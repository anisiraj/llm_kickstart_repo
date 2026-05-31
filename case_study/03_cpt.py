r"""
03_cpt.py — Continued PreTraining (CPT) of SmolLM2-135M on the chemistry corpus.

WHAT THIS DEMONSTRATES
----------------------
CPT = plain next-token prediction over raw domain text. The training signal is the **FULL causal
LM loss on every token** (no prompt to mask) — this script prints the unmasked-token fraction so you
can SEE it is ~100% (contrast with SFT in §5, which masks the prompt). We then measure **held-out
perplexity before vs after** CPT to prove the model actually absorbed the domain (lower = better).

DESIGN CONTRACT (per project rules)
-----------------------------------
- TRIAL vs FULL: top-level flag via config (env CASE_STUDY_MODE / config.set_mode). TRIAL caps the
  corpus + training steps so the whole thing runs in seconds to validate plumbing; FULL is the real run.
- Self-sufficient: if the corpus is missing it builds it (calls §1). Downloads SmolLM2-135M on first use.
- Idempotent: writes the adapter + metrics to outputs/cpt_base_<mode>/; re-running loads & reports
  instead of retraining (unless --force). Seeds are set, so a FULL retrain reproduces the numbers.
- Reproducible: records env + all settings into metrics.json.

LoRA targets include embed_tokens + lm_head (so embeddings adapt to the math-heavy distribution),
trained with a SMALLER embedding_learning_rate than the rest (see PITFALLS.md / config.CPT).

TWO BACKENDS (run the SAME CPT both ways and compare — see §8 for the head-to-head):
  • backend="hf"      — AutoModelForCausalLM + PEFT + Trainer. Runs in `.venv-rl` (trl 1.x, no unsloth).
  • backend="unsloth" — FastLanguageModel + UnslothTrainer. Runs in `.venv` (unsloth + trl 0.24).
    Unsloth must run in its own env because it pins trl/xformers (the classic two-venv split).

PACKING (the "little thing" worth documenting) — three strategies for filling sequences:
  1. NO packing: pad every example to max_len → wastes compute on padding. We don't do this.
  2. CONCATENATION packing (HF backend, manual): glue all docs together with an EOS between them and
     cut into fixed `block_size` chunks. Efficient and standard for CPT. Caveat: two documents can land
     in one block, so the model *can* attend across the EOS boundary (no per-document attention mask).
     The EOS token still signals "document boundary," which is usually enough for CPT.
  3. BEST-FIT / masked packing (Unsloth/TRL `packing=True`): packs without truncating mid-document and
     (with flash-attn varlen / reset position-ids) prevents cross-document attention. Cleaner; what the
     Unsloth backend uses. See "Fewer Truncations Improve Language Modeling" (arXiv:2404.10830).

GOTCHA — tied embeddings: SmolLM2 sets `tie_word_embeddings=True`, so `embed_tokens` and `lm_head` are
the same weights. Putting BOTH in `target_modules` adapts the tied weight (PEFT warns about merge/convert
implications). PEFT auto-sets `save_embedding_layers=True` so the adapter stores the embedding delta —
important for the GGUF export in Part B.

Run:  python case_study/03_cpt.py                              # TRIAL, HF backend (default, fast)
      python case_study/03_cpt.py --backend unsloth            # (run with .venv/bin/python)
      CASE_STUDY_MODE=full python case_study/03_cpt.py          # the real run
      python case_study/03_cpt.py --force                      # retrain even if cached
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def _load_sibling(fname: str, mod_name: str):
    """Import a digit-prefixed sibling script (e.g. 01_build_corpus.py) as a module."""
    spec = importlib.util.spec_from_file_location(mod_name, Path(__file__).resolve().parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Data: load corpus, split held-out, pack into fixed-length causal-LM blocks ─
def load_corpus_texts() -> list[str]:
    """Return the list of cleaned page texts, building the corpus first if needed (self-sufficient)."""
    manifest = config.CORPUS_DIR / "manifest.json"
    if not manifest.exists():
        print("Corpus missing — building it first (§1)...")
        _load_sibling("01_build_corpus.py", "corpus_builder").build_corpus()
    man = json.loads(manifest.read_text())
    texts = [(config.CORPUS_DIR / p["file"]).read_text() for p in man["pages"]]
    cap = config.limits()["cpt_char_cap"]
    if cap:  # TRIAL: use a small slice so the run is fast
        texts, total = [], 0
        for p in man["pages"]:
            t = (config.CORPUS_DIR / p["file"]).read_text()
            texts.append(t[:cap - total]); total += len(texts[-1])
            if total >= cap:
                break
    return texts


def pack_blocks(tokenizer, texts: list[str], block_size: int) -> list[list[int]]:
    """Concatenate documents (EOS-separated) and chunk into fixed-length blocks (standard CPT packing)."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"] + [eos])
    n = (len(ids) // block_size) * block_size
    return [ids[i:i + block_size] for i in range(0, n, block_size)]


@torch.no_grad()
def perplexity(model, blocks: list[list[int]], device) -> float:
    """Mean per-token perplexity over held-out blocks (full-sequence causal loss)."""
    model.eval()
    losses, ntok = 0.0, 0
    for b in blocks:
        ids = torch.tensor([b], device=device)
        out = model(ids, labels=ids)
        losses += out.loss.item() * (len(b) - 1)
        ntok += len(b) - 1
    return math.exp(losses / max(ntok, 1))


def run_cpt(backend: str = "hf", force: bool = False) -> dict:
    """Dispatch to the chosen CPT backend. Both produce a comparable held-out-perplexity result."""
    return (_cpt_unsloth if backend == "unsloth" else _cpt_hf)(force=force)


def _out_dir(backend: str) -> Path:
    suffix = "" if backend == "hf" else f"_{backend}"
    return config.OUTPUTS / f"cpt_base_{config.RUN_MODE}{suffix}"


def _cached(out_dir: Path, force: bool, label: str) -> dict | None:
    mp = out_dir / "metrics.json"
    if mp.exists() and not force:
        m = json.loads(mp.read_text())
        print(f"[cached] {label} CPT ({config.RUN_MODE}) done: ppl {m['ppl_before']:.2f} -> "
              f"{m['ppl_after']:.2f} ({m['ppl_delta_pct']:+.1f}%). Use --force to retrain.")
        return m
    return None


def _cpt_hf(force: bool = False) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    config.set_all_seeds()
    mode = config.RUN_MODE
    lim = config.limits()
    out_dir = _out_dir("hf")
    metrics_path = out_dir / "metrics.json"

    cached = _cached(out_dir, force, "HF")
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== §3 CPT [backend=hf, manual concatenation packing] | mode={mode} | device={device} ===")
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = load_corpus_texts()
    block = config.CPT["max_seq_len"]
    # Pack ALL text into fixed-length blocks first, THEN split blocks into train / held-out.
    # (Splitting whole documents first can starve one side when the corpus slice is small.)
    all_blocks = pack_blocks(tok, texts, block)
    n_held = min(lim["eval_blocks"], max(1, len(all_blocks) // 5))
    held_blocks, train_blocks = all_blocks[-n_held:], all_blocks[:-n_held]
    if not train_blocks:
        raise RuntimeError(f"only {len(all_blocks)} block(s) packed — raise cpt_char_cap for this mode")
    print(f"  packed: {len(train_blocks)} train blocks, {len(held_blocks)} held-out blocks "
          f"(block_size={block})")

    # Full causal loss => labels == input_ids => 0% masked. Prove it.
    ds = Dataset.from_dict({"input_ids": train_blocks,
                            "labels": [b[:] for b in train_blocks],
                            "attention_mask": [[1] * len(b) for b in train_blocks]})
    sample_labels = ds[0]["labels"]
    unmasked = sum(1 for x in sample_labels if x != -100) / len(sample_labels)
    print(f"  unmasked-token fraction = {unmasked*100:.0f}%  -> FULL causal loss (CPT), nothing masked")

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    ppl_before = perplexity(model, held_blocks, device)
    print(f"  held-out perplexity BEFORE CPT: {ppl_before:.2f}")

    lora = LoraConfig(r=config.CPT["lora_r"], lora_alpha=config.CPT["lora_alpha"],
                      lora_dropout=config.CPT["lora_dropout"],
                      target_modules=config.CPT["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Separate, smaller LR for embedding/lm_head LoRA params (PITFALLS.md trick).
    emb = [p for n, p in model.named_parameters()
           if p.requires_grad and ("embed_tokens" in n or "lm_head" in n)]
    rest = [p for n, p in model.named_parameters()
            if p.requires_grad and not ("embed_tokens" in n or "lm_head" in n)]
    opt = torch.optim.AdamW([{"params": rest, "lr": config.CPT["lr"]},
                             {"params": emb, "lr": config.CPT["embedding_learning_rate"]}],
                            weight_decay=0.0)

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=config.CPT["batch_size"],
        gradient_accumulation_steps=config.CPT["grad_accum"],
        num_train_epochs=config.CPT["epochs"] if lim["cpt_max_steps"] < 0 else 1,
        max_steps=lim["cpt_max_steps"], learning_rate=config.CPT["lr"],
        bf16=(device == "cuda"), logging_steps=2, save_strategy="no",
        report_to=[], seed=config.SEED,
    )
    Trainer(model=model, args=targs, train_dataset=ds, optimizers=(opt, None)).train()

    ppl_after = perplexity(model, held_blocks, device)
    delta = 100 * (ppl_after - ppl_before) / ppl_before
    print(f"  held-out perplexity AFTER CPT:  {ppl_after:.2f}  ({delta:+.1f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    m = dict(backend="hf", packing="manual-concatenation", mode=mode, model=config.BASE_MODEL,
             ppl_before=ppl_before, ppl_after=ppl_after, ppl_delta_pct=delta,
             train_blocks=len(train_blocks), held_blocks=len(held_blocks), block_size=block,
             unmasked_fraction=unmasked, max_steps=lim["cpt_max_steps"], lr=config.CPT["lr"],
             embedding_lr=config.CPT["embedding_learning_rate"], env=config.record_env())
    metrics_path.write_text(json.dumps(m, indent=2))
    print(f"\n  adapter + metrics saved to {out_dir}")
    print(f"  RESULT: CPT moved held-out perplexity {ppl_before:.2f} -> {ppl_after:.2f} ({delta:+.1f}%)")
    print("  (lower = the model fits domain text better). Next: §4 base-vs-instruct + forgetting.")
    # Quick visual assessment: a CPT'd base model isn't instruction-tuned, so we let it CONTINUE
    # the question text (chat=False). Eyeball whether the continuations sound domain-fluent.
    import utils
    utils.generate_samples(model, tok, n=lim["gen_samples"], chat=False,
                           title=f"§3 CPT continuations (backend=hf, mode={mode})")
    return m


def _cpt_unsloth(force: bool = False) -> dict:
    """CPT via Unsloth (run with .venv/bin/python). Uses UnslothTrainer + packing + embedding_learning_rate."""
    try:
        from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
        from transformers import AutoTokenizer
        from datasets import Dataset
    except ImportError as e:
        print(f"[skip] Unsloth backend needs the .venv env (unsloth + trl 0.24): {e}")
        print("       run:  ./.venv/bin/python case_study/03_cpt.py --backend unsloth")
        return {}

    config.set_all_seeds()
    mode, lim = config.RUN_MODE, config.limits()
    out_dir = _out_dir("unsloth")
    cached = _cached(out_dir, force, "Unsloth")
    if cached is not None:
        return cached

    print(f"=== §3 CPT [backend=unsloth, packing=True] | mode={mode} ===")
    block = config.CPT["max_seq_len"]
    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL, max_seq_length=block, dtype=None, load_in_4bit=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = load_corpus_texts()
    # Held-out blocks for an apples-to-apples perplexity comparison with the HF backend.
    all_blocks = pack_blocks(tok, texts, block)
    n_held = min(lim["eval_blocks"], max(1, len(all_blocks) // 5))
    held_blocks = all_blocks[-n_held:]
    device = next(model.parameters()).device
    ppl_before = perplexity(model, held_blocks, device)
    print(f"  held-out perplexity BEFORE CPT: {ppl_before:.2f}")

    # Unsloth feeds raw text + does best-fit packing internally.
    train_texts = texts[:-1] if len(texts) > 1 else texts
    ds = Dataset.from_dict({"text": train_texts})
    model = FastLanguageModel.get_peft_model(
        model, r=config.CPT["lora_r"], lora_alpha=config.CPT["lora_alpha"],
        lora_dropout=config.CPT["lora_dropout"], target_modules=config.CPT["target_modules"],
        use_gradient_checkpointing="unsloth", random_state=config.SEED)

    args = UnslothTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=config.CPT["batch_size"],
        gradient_accumulation_steps=config.CPT["grad_accum"],
        num_train_epochs=config.CPT["epochs"] if lim["cpt_max_steps"] < 0 else 1,
        max_steps=lim["cpt_max_steps"],
        learning_rate=config.CPT["lr"],
        embedding_learning_rate=config.CPT["embedding_learning_rate"],   # Unsloth-native smaller LR
        bf16=is_bfloat16_supported(), logging_steps=2, save_strategy="no", report_to=[], seed=config.SEED)
    trainer = UnslothTrainer(model=model, tokenizer=tok, train_dataset=ds,
                             dataset_text_field="text", max_seq_length=block,
                             packing=True, args=args)
    trainer.train()

    ppl_after = perplexity(model, held_blocks, device)
    delta = 100 * (ppl_after - ppl_before) / ppl_before
    print(f"  held-out perplexity AFTER CPT:  {ppl_after:.2f}  ({delta:+.1f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    m = dict(backend="unsloth", packing="best-fit (packing=True)", mode=mode, model=config.BASE_MODEL,
             ppl_before=ppl_before, ppl_after=ppl_after, ppl_delta_pct=delta,
             held_blocks=len(held_blocks), block_size=block, max_steps=lim["cpt_max_steps"],
             lr=config.CPT["lr"], embedding_lr=config.CPT["embedding_learning_rate"], env=config.record_env())
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\n  adapter + metrics saved to {out_dir}")
    print(f"  RESULT (unsloth): ppl {ppl_before:.2f} -> {ppl_after:.2f} ({delta:+.1f}%)")
    import utils
    utils.generate_samples(model, tok, n=lim["gen_samples"], chat=False,
                           title=f"§3 CPT continuations (backend=unsloth, mode={mode})")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Continued pretraining of SmolLM2-135M on the chem corpus.")
    ap.add_argument("--backend", choices=["hf", "unsloth"], default="hf", help="training backend")
    ap.add_argument("--force", action="store_true", help="retrain even if a cached adapter exists")
    ap.add_argument("--mode", choices=["trial", "full"], help="override TRIAL/FULL")
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run_cpt(backend=args.backend, force=args.force)


if __name__ == "__main__":
    main()
