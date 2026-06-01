r"""
07_eval.py — a consolidated EVALUATION scorecard, run before vs after SFT.

How do you know fine-tuning worked? On a 135M model you measure *relative* effects, never claim SOTA.
This section bundles the metrics used across the case study into one scorecard and runs it on the
baseline instruct model vs the SFT'd instruct model (from §5), so the improvement is visible.

METRICS (all measured, none asserted):
  • domain perplexity        — held-out chemistry text (lower = fits the domain better)
  • completion perplexity    — held-out Q&A answers given their prompts (lower = predicts answers better)
  • keyword recall (rubric)  — fraction of gold-answer content words the generation reproduces (higher better)
  • sample generations       — a few answers to eyeball

CONTRACT: TRIAL/FULL, self-sufficient (trains the §5 adapter if missing), idempotent
(caches outputs/eval_<mode>.json), reproducible.

Run:  python case_study/07_eval.py            # TRIAL
      CASE_STUDY_MODE=full python case_study/07_eval.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
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


_CPT = _sib("03_cpt.py", "cpt_for_07")    # load_corpus_texts, pack_blocks, perplexity
_S5 = _sib("05_sft.py", "s5_for_07")      # load_seed, run_sft
# completion_perplexity + keyword_recall live in utils (shared with §6; avoids an import cycle)
keyword_recall = utils.keyword_recall


def scorecard(model, tok, test, domain_blocks, device, *, label: str) -> dict:
    dom_ppl = _CPT.perplexity(model, domain_blocks, device)
    comp_ppl = utils.completion_perplexity(model, tok, test, device)
    gens = utils.generate_samples(model, tok, [p["prompt"] for p in test], chat=False, quiet=True)
    recall = sum(keyword_recall(t["completion"], g["answer"]) for t, g in zip(test, gens)) / len(test)
    print(f"  [{label}] domain ppl {dom_ppl:.2f} | completion ppl {comp_ppl:.2f} | keyword recall {recall*100:.0f}%")
    return dict(domain_ppl=dom_ppl, completion_ppl=comp_ppl, keyword_recall=recall)


def run(force: bool = False) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    config.set_all_seeds()
    lim = config.limits()
    out = config.OUTPUTS / f"eval_{config.RUN_MODE}.json"
    if out.exists() and not force:
        m = json.loads(out.read_text())
        print(f"[cached] §7 eval ({config.RUN_MODE}) — use --force to rerun.")
        print(json.dumps(m["scorecards"], indent=2))
        return m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = _S5.load_seed()
    test = rows[-config.SFT_TEST_HOLDOUT:]
    tok = AutoTokenizer.from_pretrained(config.INSTRUCT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    domain_blocks = _CPT.pack_blocks(tok, _CPT.load_corpus_texts(), config.CPT["max_seq_len"])[-lim["eval_blocks"]:]

    print(f"=== §7 eval | mode={config.RUN_MODE} | device={device} | {len(test)} test pairs ===")
    # before: untrained instruct baseline
    base = utils.load_causal_lm(config.INSTRUCT_MODEL)
    before = scorecard(base, tok, test, domain_blocks, device, label="instruct (no SFT)")
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    # after: SFT'd instruct (train via §5 if the adapter is missing — self-sufficient)
    adapter = config.OUTPUTS / f"sft_instruct_{config.RUN_MODE}" / "adapter"
    if not adapter.exists():
        print("  (no SFT adapter yet — training it via §5)")
        _S5.run_sft(init="instruct")
    m2 = PeftModel.from_pretrained(utils.load_causal_lm(config.INSTRUCT_MODEL), str(adapter))
    if not config.LOAD_IN_4BIT and device == "cuda":
        m2 = m2.to(device)
    after = scorecard(m2, tok, test, domain_blocks, device, label="instruct + SFT")

    print("\n=== before vs after SFT ===")
    print(f"  {'metric':18} | {'before':>8} | {'after':>8}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}")
    for k, lo_better in [("domain_ppl", True), ("completion_ppl", True), ("keyword_recall", False)]:
        print(f"  {k:18} | {before[k]:>8.2f} | {after[k]:>8.2f}")
    utils.generate_samples(m2, tok, [p["prompt"] for p in test], n=lim["gen_samples"], chat=False,
                           title="§7 SFT'd model — sample held-out answers")
    print("  (TRIAL barely trains, so before≈after is expected; FULL shows real movement.)")

    m = dict(mode=config.RUN_MODE, test_size=len(test),
             scorecards={"before": before, "after": after}, env=config.record_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"\n  metrics saved to {out}. Next: §8 Unsloth-vs-HF, then Part B (edge deployment).")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidated eval scorecard, before vs after SFT.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run(force=args.force)


if __name__ == "__main__":
    main()
