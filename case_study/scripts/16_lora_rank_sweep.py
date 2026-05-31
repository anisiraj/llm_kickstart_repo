r"""
16_lora_rank_sweep.py — sweep LoRA rank (with rsLoRA) and measure QUALITY vs COST.

Holds everything fixed (model, SFT data, completion-only loss, rsLoRA on) and varies only the LoRA
rank r (alpha tracks r). For each rank it SFTs the instruct model and records:
  • completion perplexity on held-out Q&A  (lower = better)
  • keyword recall on held-out generations  (higher = better)
  • trainable params, peak VRAM, wall time   (the cost side)
So you can see whether a higher rank actually buys quality, and what it costs. rsLoRA keeps high
ranks well-conditioned (scales the adapter by alpha/sqrt(r) instead of alpha/r).

Self-contained; reuses §6's trainer + utils metrics. Mutates config.SFT['lora_r'/'lora_alpha'] in
process per rank (build_sft_model reads them at call time). TRIAL caps the rank list + steps.

Run:  ../.venv/bin/python case_study/scripts/16_lora_rank_sweep.py            # SmolLM3 (4-bit)
      CASE_STUDY_MODEL=minicpm5 ../.venv/bin/python case_study/scripts/16_lora_rank_sweep.py
"""
from __future__ import annotations
try:
    import unsloth  # noqa: F401  (import first for the 4-bit path)
except Exception:
    pass
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402


def _sib(fname, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_S5 = _sib("05_sft.py", "s5_for_16")   # load_seed
_S6 = _sib("06_base_vs_instruct_sweep.py", "s6_for_16")   # sft_train (reads config.SFT at call time)

RANKS = [8, 32] if config.is_trial() else [8, 16, 32, 64, 128]


def run() -> dict:
    config.set_all_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lim = config.limits()
    rows = _S5.load_seed()
    test = rows[-config.SFT_TEST_HOLDOUT:]
    pool = rows[:-config.SFT_TEST_HOLDOUT]
    print(f"=== §16 LoRA-rank sweep | model={config.MODEL_KEY} mode={config.RUN_MODE} | "
          f"rsLoRA=True | ranks={RANKS} ===")
    print(f"  fixed: instruct model, {len(pool)} SFT pairs, completion-only loss; vary r (alpha=r)\n")

    results = []
    for r in RANKS:
        config.SFT["lora_r"] = config.SFT["lora_alpha"] = r    # in-process override (rsLoRA stays on)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, tok = _S6.sft_train(config.INSTRUCT_MODEL, pool, device, lim)
        dt = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        comp = utils.completion_perplexity(model, tok, test, device)
        gens = utils.generate_samples(model, tok, [p["prompt"] for p in test],
                                      chat=False, quiet=True, max_new_tokens=64)
        recall = sum(utils.keyword_recall(t["completion"], g["answer"])
                     for t, g in zip(test, gens)) / len(test)
        row = dict(rank=r, completion_ppl=comp, recall=recall, trainable=trainable,
                   vram_gb=vram, seconds=dt)
        results.append(row)
        print(f"  r={r:>4} | completion ppl {comp:8.2f} | recall {recall*100:4.0f}% | "
              f"trainable {trainable/1e6:6.1f}M | VRAM {vram:5.2f}GB | {dt:6.1f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n  rank | completion ppl | recall | trainable | VRAM | time")
    print("  -----+----------------+--------+-----------+------+------")
    for x in results:
        print(f"  {x['rank']:>4} | {x['completion_ppl']:>14.2f} | {x['recall']*100:>5.0f}% | "
              f"{x['trainable']/1e6:>8.1f}M | {x['vram_gb']:>4.2f} | {x['seconds']:>5.0f}s")
    best = min(results, key=lambda x: x["completion_ppl"])
    print(f"\n  Lowest completion ppl at r={best['rank']} ({best['completion_ppl']:.2f}). "
          "Watch for diminishing returns: higher r costs params/VRAM/time for often-small quality gains.")

    out = {"model": config.MODEL_KEY, "mode": config.RUN_MODE, "use_rslora": True, "results": results}
    (config.OUTPUTS / f"lora_sweep_{config.RUN_MODE}.json").write_text(json.dumps(out, indent=2))
    print(f"  saved -> {config.OUTPUTS / f'lora_sweep_{config.RUN_MODE}.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trial", "full"])
    a = ap.parse_args()
    if a.mode:
        config.set_mode(a.mode)
    run()
