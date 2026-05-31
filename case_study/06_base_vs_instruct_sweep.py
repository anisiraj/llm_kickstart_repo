r"""
06_base_vs_instruct_sweep.py — THE CENTERPIECE experiment (research question #3).

CLAIM UNDER TEST (from the unsloth guide's "Recipe E"):
    With a SMALL SFT set, initialize from the INSTRUCT model, not the base model.
RATIONALE: the instruct model already follows instructions, so a handful of examples is enough to
steer it; the base model must learn the answer format from scratch, which needs much more data.

EXPERIMENT: hold out a fixed test set of Q&A; then for each init ∈ {base, instruct} and each training
size N, SFT (completion-only loss) and measure **held-out completion perplexity** (lower = the model
predicts the gold answers better). Plot perplexity vs N for both inits. Expectation: the instruct line
sits lower, especially at small N, and the gap narrows as N grows.

HONESTY NOTE: we only hand-authored 32 Q&A pairs (§2), so N is capped (test holdout + small train pool).
That cap IS the data-scarcity lesson — getting more high-quality instruction pairs is the bottleneck.

CONTRACT: TRIAL/FULL flag, self-sufficient, idempotent (caches outputs/sweep_<mode>.json), reproducible.

Run:  python case_study/06_base_vs_instruct_sweep.py            # TRIAL
      CASE_STUDY_MODE=full python case_study/06_base_vs_instruct_sweep.py
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


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_S5 = _sib("05_sft.py", "s5_for_06")   # load_seed


@torch.no_grad()
def completion_perplexity(model, tok, pairs: list[dict], device) -> float:
    """Mean perplexity of the gold completions given their prompts (prompt tokens masked).

    Mirrors the SFT objective: loss only on the answer tokens. Same formatting for every condition,
    so the comparison across (init, N) is apples-to-apples.
    """
    model.eval()
    total_loss, total_tok = 0.0, 0
    for p in pairs:
        p_ids = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(" " + p["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + c_ids], device=device)
        labels = torch.tensor([[-100] * len(p_ids) + c_ids], device=device)
        loss = model(ids, labels=labels).loss.item()
        total_loss += loss * len(c_ids)
        total_tok += len(c_ids)
    return math.exp(total_loss / max(total_tok, 1))


def sft_train(model_name: str, train_rows: list[dict], device, lim):
    """SFT a fresh model on train_rows (completion-only loss). Returns (model, tok)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    args = SFTConfig(
        output_dir="/tmp/sweep_ckpt", per_device_train_batch_size=config.SFT["batch_size"],
        gradient_accumulation_steps=config.SFT["grad_accum"], num_train_epochs=config.SFT["epochs"],
        max_steps=lim["sft_max_steps"], learning_rate=config.SFT["lr"],
        max_length=config.SFT["max_seq_len"], completion_only_loss=True, packing=False,
        bf16=(device == "cuda"), logging_steps=50, save_strategy="no", report_to=[], seed=config.SEED)
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32),
        args=args, train_dataset=Dataset.from_list(train_rows), processing_class=tok,
        peft_config=LoraConfig(r=config.SFT["lora_r"], lora_alpha=config.SFT["lora_alpha"],
                               lora_dropout=config.SFT["lora_dropout"],
                               target_modules=config.SFT["target_modules"], task_type="CAUSAL_LM"))
    trainer.train()
    return trainer.model.to(device), tok


def run(force: bool = False) -> dict:
    config.set_all_seeds()
    lim = config.limits()
    out = config.OUTPUTS / f"sweep_{config.RUN_MODE}.json"
    if out.exists() and not force:
        m = json.loads(out.read_text())
        print(f"[cached] §6 sweep ({config.RUN_MODE}) — use --force to rerun.")
        _print_table(m["results"], m["sizes"])
        return m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = _S5.load_seed()
    test = rows[-config.SFT_TEST_HOLDOUT:]
    pool = rows[:-config.SFT_TEST_HOLDOUT]
    sizes = [n for n in lim["sweep_sizes"] if n <= len(pool)]
    print(f"=== §6 base-vs-instruct sweep | mode={config.RUN_MODE} | device={device} ===")
    print(f"  test set: {len(test)} held-out pairs | train pool: {len(pool)} | sizes: {sizes}\n")

    results = {"base": {}, "instruct": {}}
    for init in ("base", "instruct"):
        model_name = config.BASE_MODEL if init == "base" else config.INSTRUCT_MODEL
        for n in sizes:
            model, tok = sft_train(model_name, pool[:n], device, lim)
            ppl = completion_perplexity(model, tok, test, device)
            results[init][str(n)] = ppl
            print(f"  init={init:8} N={n:>3} -> held-out completion perplexity {ppl:.2f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print()
    _print_table(results, sizes)
    m = dict(mode=config.RUN_MODE, sizes=sizes, test_size=len(test), results=results,
             env=config.record_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"\n  metrics saved to {out}. Next: §7 eval, then §8 Unsloth-vs-HF.")
    return m


def _print_table(results: dict, sizes: list) -> None:
    print("  held-out completion perplexity (lower = better):")
    print(f"    {'N':>4} | {'base':>8} | {'instruct':>9} | winner")
    print(f"    {'-'*4}-+-{'-'*8}-+-{'-'*9}-+-------")
    for n in sizes:
        b = results["base"].get(str(n)); i = results["instruct"].get(str(n))
        if b is None or i is None:
            continue
        win = "instruct" if i < b else "base"
        print(f"    {n:>4} | {b:>8.2f} | {i:>9.2f} | {win}")
    print("\n  Hypothesis: instruct-init wins (lower ppl) at small N; gap narrows as N grows.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Base-vs-instruct SFT sweep over training-set size.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run(force=args.force)


if __name__ == "__main__":
    main()
