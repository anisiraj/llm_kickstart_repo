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
# Import unsloth FIRST (before torch/transformers/trl) for clean QLoRA patching + dataset pickling.
# No-op in .venv-rl (unsloth not installed).
try:
    import unsloth  # noqa: F401,E402
except Exception:
    pass
import argparse
import importlib.util
import json
import math
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


_S5 = _sib("05_sft.py", "s5_for_06")   # load_seed
completion_perplexity = utils.completion_perplexity   # shared (lives in utils to avoid import cycle)


def sft_train(model_name: str, train_rows: list[dict], device, lim):
    """SFT a fresh model on train_rows (completion-only loss). Returns (model, tok)."""
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    sft_model, tok = utils.build_sft_model(model_name)   # bf16 (135M) or 4-bit QLoRA via Unsloth
    args = SFTConfig(
        output_dir="/tmp/sweep_ckpt", per_device_train_batch_size=config.SFT["batch_size"],
        gradient_accumulation_steps=config.SFT["grad_accum"], num_train_epochs=config.SFT["epochs"],
        max_steps=lim["sft_max_steps"], learning_rate=config.SFT["lr"],
        max_length=config.SFT["max_seq_len"], completion_only_loss=True, packing=False, dataset_num_proc=1,
        bf16=(device == "cuda"), logging_steps=50, save_strategy="no", report_to=[], seed=config.SEED)
    trainer = SFTTrainer(model=sft_model, args=args,
                         train_dataset=Dataset.from_list(train_rows), processing_class=tok)
    trainer.train()
    model = trainer.model
    return (model if config.LOAD_IN_4BIT else model.to(device)), tok


def run(force: bool = False) -> dict:
    config.set_all_seeds()
    lim = config.limits()
    out = config.OUTPUTS / f"sweep_{config.RUN_MODE}.json"
    if out.exists() and not force:
        m = json.loads(out.read_text())
        print(f"[cached] §6 sweep ({config.RUN_MODE}) — use --force to rerun.")
        _print_table(m["results"], m.get("recall", {"base": {}, "instruct": {}}), m["sizes"])
        return m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = _S5.load_seed()
    test = rows[-config.SFT_TEST_HOLDOUT:]
    pool = rows[:-config.SFT_TEST_HOLDOUT]
    sizes = [n for n in lim["sweep_sizes"] if n <= len(pool)]
    print(f"=== §6 base-vs-instruct sweep | mode={config.RUN_MODE} | device={device} ===")
    print(f"  test set: {len(test)} held-out pairs | train pool: {len(pool)} | sizes: {sizes}\n")

    import utils
    results = {"base": {}, "instruct": {}}        # completion perplexity (format-sensitive)
    recall = {"base": {}, "instruct": {}}         # keyword recall on generations (format-robust)
    for init in ("base", "instruct"):
        model_name = config.BASE_MODEL if init == "base" else config.INSTRUCT_MODEL
        for n in sizes:
            model, tok = sft_train(model_name, pool[:n], device, lim)
            ppl = completion_perplexity(model, tok, test, device)
            gens = utils.generate_samples(model, tok, [p["prompt"] for p in test],
                                          chat=False, quiet=True, max_new_tokens=64)
            rec = sum(utils.keyword_recall(t["completion"], g["answer"])
                      for t, g in zip(test, gens)) / len(test)
            results[init][str(n)] = ppl
            recall[init][str(n)] = rec
            print(f"  init={init:8} N={n:>3} -> completion ppl {ppl:.2f} | keyword recall {rec*100:.0f}%")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print()
    _print_table(results, recall, sizes)
    m = dict(mode=config.RUN_MODE, sizes=sizes, test_size=len(test), results=results,
             recall=recall, env=config.record_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"\n  metrics saved to {out}. Next: §7 eval, then §8 Unsloth-vs-HF.")
    return m


def _print_table(results: dict, recall: dict, sizes: list) -> None:
    print("  Two metrics per N — completion perplexity (lower=better) and keyword recall (higher=better):")
    print(f"    {'N':>4} | {'base ppl':>8} {'inst ppl':>8} | {'base rec':>8} {'inst rec':>8} | winner(recall)")
    print(f"    {'-'*4}-+-{'-'*17}-+-{'-'*17}-+--------------")
    for n in sizes:
        b, i = results["base"].get(str(n)), results["instruct"].get(str(n))
        br, ir = recall["base"].get(str(n)), recall["instruct"].get(str(n))
        if b is None or i is None:
            continue
        win = "instruct" if (ir or 0) > (br or 0) else "base"
        print(f"    {n:>4} | {b:>8.2f} {i:>8.2f} | {br*100:>7.0f}% {ir*100:>7.0f}% | {win}")
    print("\n  Note: completion perplexity is format-sensitive (the chat-tuned instruct model is penalized")
    print("  on plain prompt-completion text), so KEYWORD RECALL on generations is the fairer signal.")
    print("  Hypothesis: instruct-init reaches usable answers (higher recall) with fewer examples.")


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
