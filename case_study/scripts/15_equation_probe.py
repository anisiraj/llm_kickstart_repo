r"""
15_equation_probe.py — did feeding LaTeX into CPT teach the model to REPRODUCE domain equations?

The corpus (§1) preserved ~1,150 equations as inline `$LaTeX$`, and CPT (§3) trained the embeddings
(`embed_tokens`+`lm_head` are in the LoRA targets) on that math-heavy stream. This probe closes the
loop: we ask the model to WRITE canonical chemistry/physics equations in LaTeX and score how many of
the characteristic LaTeX tokens appear — comparing the plain base model vs the CPT'd model.

It is fully self-contained (our own models, via Unsloth/PEFT — no external API). Honest expectation:
a strong base (SmolLM3) may already know textbook equations, so CPT's gain can be small; a weaker base
(135M) should improve more. Either way the number is measured, not asserted.

Run:  ../.venv/bin/python case_study/15_equation_probe.py        # SmolLM3 (4-bit)
      python case_study/15_equation_probe.py                     # 135M (bf16)
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

# Canonical equations + the LaTeX tokens that characterize a correct rendering (case-insensitive).
EQUATION_PROBES = [
    dict(name="time-dependent Schrödinger equation",
         prompt="The time-dependent Schrödinger equation, in LaTeX, is:",
         keys=[r"\hbar", r"\partial", r"\psi", "="]),
    dict(name="time-independent Schrödinger equation",
         prompt="The time-independent Schrödinger equation, in LaTeX, is:",
         keys=[r"\hat", "h", r"\psi", "e", "="]),
    dict(name="kinetic energy operator",
         prompt="The quantum kinetic energy operator, in LaTeX, is:",
         keys=[r"\hbar", r"\nabla", "2", "m"]),
    dict(name="DFT total energy functional",
         prompt="In density functional theory the total energy as a functional of the density, in LaTeX, is:",
         keys=[r"\rho", "e", "["]),
]


def _load(model_name: str, adapter: str | None):
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


def _score(text: str, keys: list[str]) -> float:
    t = text.lower()
    return sum(1 for k in keys if k.lower() in t) / len(keys)


def probe(model, tok, label: str) -> dict:
    gens = utils.generate_samples(model, tok, [e["prompt"] for e in EQUATION_PROBES],
                                  chat=False, quiet=True, max_new_tokens=80)
    scores, shown = [], []
    for e, g in zip(EQUATION_PROBES, gens):
        s = _score(g["answer"], e["keys"])
        scores.append(s)
        shown.append({"eq": e["name"], "score": s, "gen": g["answer"][:140]})
    avg = sum(scores) / len(scores)
    print(f"  [{label}] LaTeX-token reproduction: {avg*100:.0f}%")
    for sh in shown:
        print(f"      {sh['eq']:38} {sh['score']*100:3.0f}%  {sh['gen']!r}")
    return {"label": label, "avg": avg, "detail": shown}


def run() -> dict:
    config.set_all_seeds()
    mode = config.RUN_MODE
    cpt_ad = _adapter(f"cpt_base_{mode}_unsloth") or _adapter(f"cpt_base_{mode}")
    print(f"=== §15 equation reproduction | model={config.MODEL_KEY} mode={mode} ===")
    print("  Did CPT on the LaTeX-rich corpus (with embeddings retrained) teach equation generation?")
    out = {"mode": mode, "model": config.MODEL_KEY, "stages": {}}

    m, t = _load(config.BASE_MODEL, None)
    out["stages"]["base"] = probe(m, t, "base")
    del m
    torch.cuda.is_available() and torch.cuda.empty_cache()

    if cpt_ad:
        m, t = _load(config.BASE_MODEL, cpt_ad)
        out["stages"]["base+CPT"] = probe(m, t, "base+CPT")
        del m
        torch.cuda.is_available() and torch.cuda.empty_cache()
        b, c = out["stages"]["base"]["avg"], out["stages"]["base+CPT"]["avg"]
        delta = (c - b) * 100
        print(f"\n  RESULT: base {b*100:.0f}% -> base+CPT {c*100:.0f}%  ({delta:+.0f} pts)")
        print("  (CPT on a strong base may move this little — it already knows textbook equations;")
        print("   a weaker base should gain more. The corpus + retrained embeddings are what enable it.)")
    else:
        print("  [no CPT adapter] run §3 first to compare base vs base+CPT.")

    (config.OUTPUTS / "equation_probe.json").write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {config.OUTPUTS/'equation_probe.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trial", "full"])
    a = ap.parse_args()
    if a.mode:
        config.set_mode(a.mode)
    run()
