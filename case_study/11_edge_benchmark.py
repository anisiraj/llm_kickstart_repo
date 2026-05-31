r"""
11_edge_benchmark.py — Part B step 3: measure the on-device footprint and speed.

The whole reason to fine-tune a 135M model is that it runs on small hardware. This measures, via the
deployed Ollama model:
  • FOOTPRINT — the quantized model size on disk (a good proxy for RAM needed to serve it)
  • SPEED     — local generation tokens/sec (num_predict forces enough tokens to time it)

WHY THIS IS THE EDGE STORY (research-backed, see RESEARCH_NOTES §5): a Raspberry Pi 5 runs 1–3B models
at ~10–18 tok/s and TinyLlama-1.1B-Q4 at ~8–12 tok/s on a Pi 4. A 135M model is far smaller/faster and
needs only tens of MB — comfortably real-time on a Pi/phone-class board. To reproduce on a Pi: install
Ollama there, `ollama create` the same Modelfile, and hit the same `/api/generate` endpoint — the code
below is identical; only the hardware changes.

CONTRACT: self-sufficient (deploys via §10 if needed), idempotent, graceful skip without Ollama.

Run:  python case_study/11_edge_benchmark.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OLLAMA_URL = "http://localhost:11434"


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _model_footprint(name: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except Exception:
        return None
    for m in tags.get("models", []):
        if m["name"].startswith(name):
            det = m.get("details", {})
            return {"size_gb": m.get("size", 0) / 1e9,
                    "params": det.get("parameter_size", "?"), "quant": det.get("quantization_level", "?")}
    return None


def _speed(name: str, n_predict: int = 128) -> dict:
    body = json.dumps({"model": name, "prompt": "Explain what a basis set is, in detail.",
                       "stream": False, "options": {"num_predict": n_predict, "temperature": 0.7}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    toks, secs = d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9
    return {"tokens": toks, "tok_per_s": (toks / secs) if (secs > 0 and toks >= 8) else None}


def run() -> dict:
    import shutil
    config.set_all_seeds()
    print(f"=== §11 edge benchmark | mode={config.RUN_MODE} ===")
    if not shutil.which("ollama"):
        print("  [skip] no `ollama` — see §10. (On a Pi, install Ollama and re-run this same script.)")
        return {"skipped": "no ollama"}

    name = f"chem-sft-{config.RUN_MODE}"
    fp = _model_footprint(name)
    if fp is None:
        print("  (model not deployed — deploying via §10)")
        if _sib("10_ollama_deploy.py", "s10_for_11").run().get("skipped"):
            return {"skipped": "deploy failed"}
        fp = _model_footprint(name)

    print(f"  FOOTPRINT: {fp['size_gb']:.3f} GB on disk | params {fp['params']} | quant {fp['quant']}")
    sp = _speed(name)
    if sp["tok_per_s"] is not None:
        print(f"  SPEED: {sp['tok_per_s']:.1f} tok/s on {config.record_env()['cuda']} ({sp['tokens']} tokens)")
    else:
        print(f"  SPEED: N/A (model generated {sp['tokens']} tokens — degenerate TRIAL model; "
              "use a FULL fine-tune for a real number)")

    print("\n  REFERENCE (research, RESEARCH_NOTES §5): Raspberry Pi 5 runs 1-3B models at ~10-18 tok/s;")
    print("  a 135M model is far lighter (tens of MB RAM) and faster -> genuinely real-time on edge HW.")
    print("  Reproduce on a Pi: install Ollama, `ollama create` the same Modelfile, hit /api/generate.")
    m = dict(mode=config.RUN_MODE, model=name, footprint=fp, speed=sp, gpu=config.record_env()["cuda"])
    (config.OUTPUTS / f"edge_{config.RUN_MODE}.json").write_text(json.dumps(m, indent=2))
    print("\n  Next: §12 — evaluate with lm-evaluation-harness + run the model as an agent.")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge footprint + speed benchmark via Ollama.")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run()


if __name__ == "__main__":
    main()
