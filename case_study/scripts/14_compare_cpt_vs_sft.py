r"""
14_compare_cpt_vs_sft.py — CPT-only vs SFT, side by side (visual + numerical). The money shot.

This makes the difference between the two training phases concrete on the SAME prompts:

  • "after CPT only"  = base model + the §3 CPT adapter. CPT is plain next-token prediction over raw
    domain text with **FULL causal loss on every token** (no prompt masking). Result: the model is
    domain-fluent but NOT instruction-tuned — asked a question, it CONTINUES like an article.
  • "after SFT"       = instruct model + the §5 SFT adapter. SFT trains on prompt→completion pairs with
    loss on the **COMPLETION only** (the prompt is masked). Result: the model ANSWERS the question.

WHERE THE LOSS DIFFERENCE IS SET (the thing to point at in the book):
  • CPT  (§3): labels == input_ids, nothing masked  -> unmasked fraction ~100%   (full causal loss)
  • SFT  (§5/§6): TRL `SFTConfig(completion_only_loss=True)` -> prompt tokens become -100 (~30-80% unmasked)
Run settings come straight from config.CPT / config.SFT (printed below): CPT = 1 epoch, LR 5e-5,
embedding_learning_rate 5e-6, targets incl. embed_tokens+lm_head; SFT = 3 epochs, LR 2e-4.

CONTRACT: inference-only over the adapters trained earlier (run §3 + §5 first, or the full pipeline).
Graceful skip if an adapter is missing. Writes outputs/<model>/compare.json for the chapter.

Run:  python case_study/14_compare_cpt_vs_sft.py            # (use ../.venv/bin/python for SmolLM3)
"""
from __future__ import annotations
try:
    import unsloth  # noqa: F401  (import first for the 4-bit path)
except Exception:
    pass
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

_CPT = importlib.util.spec_from_file_location("cpt14", Path(__file__).resolve().parent / "03_cpt.py")
_CPTM = importlib.util.module_from_spec(_CPT); _CPT.loader.exec_module(_CPTM)
_S5 = importlib.util.spec_from_file_location("s5_14", Path(__file__).resolve().parent / "05_sft.py")
_S5M = importlib.util.module_from_spec(_S5); _S5.loader.exec_module(_S5M)

SELECT = config.SAMPLE_QUESTIONS[:3]   # the prompts we show side by side


def _load(model_name: str, adapter: str | None):
    """Load model (+ optional LoRA adapter) for inference, 4-bit via Unsloth or bf16 via HF."""
    if config.LOAD_IN_4BIT:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=config.CPT["max_seq_len"], dtype=None, load_in_4bit=True)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        FastLanguageModel.for_inference(model)
        return model, tok
    from transformers import AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = utils.load_causal_lm(model_name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def _adapter(subdir: str) -> str | None:
    p = config.OUTPUTS / subdir / "adapter"
    return str(p) if (p / "adapter_config.json").exists() else None


def run() -> dict:
    config.set_all_seeds()
    mode = config.RUN_MODE
    cpt_ad = _adapter(f"cpt_base_{mode}_unsloth") or _adapter(f"cpt_base_{mode}")
    sft_ad = _adapter(f"sft_instruct_{mode}")
    print(f"=== §14 CPT-only vs SFT — side by side | model={config.MODEL_KEY} mode={mode} ===")
    print(f"  CPT settings: {config.CPT['epochs']} epoch, lr={config.CPT['lr']}, "
          f"embed_lr={config.CPT['embedding_learning_rate']}, FULL causal loss (no mask)")
    print(f"  SFT settings: {config.SFT['epochs']} epochs, lr={config.SFT['lr']}, "
          f"completion_only_loss=True (prompt masked)")
    if not cpt_ad or not sft_ad:
        print(f"  [skip] need both adapters — CPT:{bool(cpt_ad)} SFT:{bool(sft_ad)}. Run §3 and §5 first.")
        return {"skipped": True}

    # held-out domain blocks for perplexity (built per tokenizer at use time)
    texts = _CPTM.load_corpus_texts()
    test = _S5M.load_seed()[-config.SFT_TEST_HOLDOUT:]
    out = {"mode": mode, "model": config.MODEL_KEY, "stages": {}, "generations": {}}

    stages = [
        ("base",          config.BASE_MODEL,     None,   False),
        ("base+CPT",      config.BASE_MODEL,     cpt_ad, False),
        ("instruct",      config.INSTRUCT_MODEL, None,   True),
        ("instruct+SFT",  config.INSTRUCT_MODEL, sft_ad, True),
    ]
    for tag, name, adapter, is_instruct in stages:
        model, tok = _load(name, adapter)
        dev = next(model.parameters()).device
        blocks = _CPTM.pack_blocks(tok, texts, config.CPT["max_seq_len"])[-config.limits()["eval_blocks"]:]
        ppl = _CPTM.perplexity(model, blocks, dev)
        comp = utils.completion_perplexity(model, tok, test, dev)
        gens = utils.generate_samples(model, tok, SELECT, chat=False, quiet=True, max_new_tokens=70)
        # recall is scored on the TEST prompts (must generate those, not the SELECT display prompts)
        test_gens = utils.generate_samples(model, tok, [p["prompt"] for p in test],
                                           chat=False, quiet=True, max_new_tokens=70)
        recall = sum(utils.keyword_recall(t["completion"], g["answer"])
                     for t, g in zip(test, test_gens)) / len(test)
        out["stages"][tag] = dict(domain_ppl=ppl, completion_ppl=comp, keyword_recall=recall)
        out["generations"][tag] = gens
        print(f"  {tag:14} domain ppl {ppl:7.2f} | completion ppl {comp:8.2f} | recall {recall*100:4.0f}%")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── visual side-by-side: CPT-only (continuation) vs SFT (answer) ──
    print("\n=== VISUAL: same prompt, after CPT-only vs after SFT ===")
    for i, q in enumerate(SELECT):
        print(f"\nQ: {q}")
        print(f"  [after CPT only] {out['generations']['base+CPT'][i]['answer'][:200]!r}")
        print(f"  [after SFT]      {out['generations']['instruct+SFT'][i]['answer'][:200]!r}")
    print("\n  CPT-only continues/expands the text (domain LM); SFT answers the question (instruction-tuned).")
    print("  Numerically: CPT lowers DOMAIN perplexity; SFT lowers COMPLETION perplexity + raises recall.")

    (config.OUTPUTS / "compare.json").write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {config.OUTPUTS/'compare.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trial", "full"])
    a = ap.parse_args()
    if a.mode:
        config.set_mode(a.mode)
    run()
