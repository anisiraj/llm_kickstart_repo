r"""
04_cpt_base_vs_instruct.py — CPT from a BASE vs an INSTRUCT model, and a forgetting smoke test.

RESEARCH QUESTION #4: when you continue-pretrain on a niche domain, does it matter whether you start
from the base model or the instruct model — and does CPT damage general ability ("catastrophic
forgetting")?

WHAT WE MEASURE (so the answer is shown, not asserted)
------------------------------------------------------
For each starting point (base, instruct) we record, before and after CPT on the chemistry corpus:
  • DOMAIN perplexity on held-out chemistry text  -> should DROP  (the model is learning the domain)
  • GENERAL perplexity on held-out non-chemistry text -> if it RISES, that is forgetting
  • a few GENERAL question generations -> eyeball whether instruction-following degraded
The instruct model has more to lose (it can follow instructions); the base model has little general
"skill" to forget. Mitigations (lower LR, ≤1 epoch, ~10% replay) are in PITFALLS.md / RESEARCH_NOTES.

CONTRACT: TRIAL/FULL flag, self-sufficient (builds domain corpus + fetches a small general corpus if
missing, downloads both models), idempotent (caches outputs/forgetting_<mode>.json), reproducible.

Run:  python case_study/04_cpt_base_vs_instruct.py            # TRIAL (default)
      CASE_STUDY_MODE=full python case_study/04_cpt_base_vs_instruct.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_CB = _sib("01_build_corpus.py", "cb_for_04")     # fetch_plaintext, clean, slugify
_CPT = _sib("03_cpt.py", "cpt_for_04")            # load_corpus_texts, pack_blocks, perplexity


def general_corpus_texts() -> list[str]:
    """Fetch + cache a small NON-chemistry corpus for the forgetting metric (self-sufficient)."""
    gdir = config.DATA / "general"
    gdir.mkdir(parents=True, exist_ok=True)
    texts = []
    for title in config.GENERAL_PAGES:
        f = gdir / f"{_CB.slugify(title)}.txt"
        if not f.exists():
            resolved, raw = _CB.fetch_plaintext(title)
            txt, _ = _CB.clean(raw)
            f.write_text(txt)
        texts.append(f.read_text())
    return texts


def _train_cpt(model, lim):
    """Attach LoRA (incl. embeddings, split LR) and run CPT. Returns the trained PEFT model."""
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=config.CPT["lora_r"], lora_alpha=config.CPT["lora_alpha"],
        lora_dropout=config.CPT["lora_dropout"], use_rslora=config.CPT["use_rslora"],
        target_modules=config.CPT["target_modules"], task_type="CAUSAL_LM"))
    emb = [p for n, p in model.named_parameters()
           if p.requires_grad and ("embed_tokens" in n or "lm_head" in n)]
    rest = [p for n, p in model.named_parameters()
            if p.requires_grad and not ("embed_tokens" in n or "lm_head" in n)]
    opt = torch.optim.AdamW([{"params": rest, "lr": config.CPT["lr"]},
                             {"params": emb, "lr": config.CPT["embedding_learning_rate"]}])
    return model, opt


def run_one(model_name: str, tag: str, is_instruct: bool, domain_texts, general_texts, lim) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block = config.CPT["max_seq_len"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # held-out blocks (per this model's tokenizer)
    dom_all = _CPT.pack_blocks(tok, domain_texts, block)
    n_held = min(lim["eval_blocks"], max(1, len(dom_all) // 5))
    dom_held, dom_train = dom_all[-n_held:], dom_all[:-n_held]
    gen_held = _CPT.pack_blocks(tok, general_texts, block)[: lim["eval_blocks"]]

    print(f"\n=== {tag} ({model_name}) ===")
    model = utils.load_causal_lm(model_name, training=True)   # 4-bit QLoRA if active model needs it
    dom_before = _CPT.perplexity(model, dom_held, device)
    gen_before = _CPT.perplexity(model, gen_held, device)
    print(f"  before CPT: domain ppl {dom_before:.2f} | general ppl {gen_before:.2f}")
    utils.generate_samples(model, tok, config.GENERAL_QUESTIONS, n=lim["gen_samples"],
                           chat=is_instruct, title=f"{tag} GENERAL answers BEFORE CPT")

    model, opt = _train_cpt(model, lim)
    ds = Dataset.from_dict({"input_ids": dom_train, "labels": [b[:] for b in dom_train],
                            "attention_mask": [[1] * len(b) for b in dom_train]})
    targs = TrainingArguments(
        output_dir=str(config.OUTPUTS / f"cpt_{tag}_{config.RUN_MODE}" / "ckpt"),
        per_device_train_batch_size=config.CPT["batch_size"],
        gradient_accumulation_steps=config.CPT["grad_accum"],
        num_train_epochs=1, max_steps=lim["cpt_max_steps"], learning_rate=config.CPT["lr"],
        gradient_checkpointing=config.GRAD_CKPT,           # essential for a 3B in 4-bit on 12GB
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.GRAD_CKPT else None,
        bf16=(device == "cuda"), logging_steps=5, save_strategy="no", report_to=[], seed=config.SEED)
    Trainer(model=model, args=targs, train_dataset=ds, optimizers=(opt, None)).train()

    dom_after = _CPT.perplexity(model, dom_held, device)
    gen_after = _CPT.perplexity(model, gen_held, device)
    dom_delta = 100 * (dom_after - dom_before) / dom_before
    gen_delta = 100 * (gen_after - gen_before) / gen_before
    print(f"  after CPT:  domain ppl {dom_after:.2f} ({dom_delta:+.1f}%) | "
          f"general ppl {gen_after:.2f} ({gen_delta:+.1f}%  <- forgetting if positive)")
    utils.generate_samples(model, tok, config.GENERAL_QUESTIONS, n=lim["gen_samples"],
                           chat=is_instruct, title=f"{tag} GENERAL answers AFTER CPT")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(tag=tag, model=model_name, is_instruct=is_instruct,
                domain_ppl_before=dom_before, domain_ppl_after=dom_after, domain_delta_pct=dom_delta,
                general_ppl_before=gen_before, general_ppl_after=gen_after, general_delta_pct=gen_delta)


def run(force: bool = False) -> dict:
    config.set_all_seeds()
    lim = config.limits()
    out = config.OUTPUTS / f"forgetting_{config.RUN_MODE}.json"
    if out.exists() and not force:
        m = json.loads(out.read_text())
        print(f"[cached] §4 forgetting ({config.RUN_MODE}) — use --force to rerun.")
        print(json.dumps(m["summary"], indent=2))
        return m

    domain_texts = _CPT.load_corpus_texts()
    general_texts = general_corpus_texts()
    print(f"=== §4 base-vs-instruct CPT + forgetting | mode={config.RUN_MODE} ===")
    base = run_one(config.BASE_MODEL, "base", False, domain_texts, general_texts, lim)
    inst = run_one(config.INSTRUCT_MODEL, "instruct", True, domain_texts, general_texts, lim)

    print("\n=== SUMMARY: domain learning vs general forgetting ===")
    print(f"  {'start':9} | {'domain ppl Δ':>13} | {'general ppl Δ':>14}")
    print(f"  {'-'*9}-+-{'-'*13}-+-{'-'*14}")
    for r in (base, inst):
        print(f"  {r['tag']:9} | {r['domain_delta_pct']:>+12.1f}% | {r['general_delta_pct']:>+13.1f}%")
    print("\n  Reading it: domain Δ negative = learned the domain; general Δ positive = forgetting.")
    print("  Expectation: the INSTRUCT start has more general ability to lose, so watch its general Δ")
    print("  and its AFTER answers above. Mitigate with lower LR, ≤1 epoch, and ~10% replay (PITFALLS.md).")

    m = dict(mode=config.RUN_MODE, base=base, instruct=inst,
             summary={"base_domain_delta_pct": base["domain_delta_pct"],
                      "base_general_delta_pct": base["general_delta_pct"],
                      "instruct_domain_delta_pct": inst["domain_delta_pct"],
                      "instruct_general_delta_pct": inst["general_delta_pct"]},
             env=config.record_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"\n  metrics saved to {out}. Next: §5 SFT (completion-only loss).")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="CPT base-vs-instruct + catastrophic-forgetting smoke test.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run(force=args.force)


if __name__ == "__main__":
    main()
