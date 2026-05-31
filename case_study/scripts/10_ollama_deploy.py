r"""
10_ollama_deploy.py — Part B step 2: deploy the fine-tuned model with Ollama and query it.

Ollama is the easy on-device runtime: a Modelfile points at weights, `ollama create` imports/quantizes
them, and a local REST API (port 11434, OpenAI-compatible at /v1/chat/completions) serves the model.
Ollama can import Llama-architecture **safetensors** directly, so it works on the §9 merged model even
without a GGUF (it'll use the GGUF if §9 produced one — the Modelfile's FROM handles both).

This also measures **local tokens/sec** (eval_count / eval_duration from the API) — the metric that
matters at the edge. (§11 documents how the same number is taken on a Raspberry Pi.)

CONTRACT: self-sufficient (merges via §9 if needed), idempotent (skips create if the model exists),
gracefully skips if the `ollama` binary/server isn't available.

Run:  python case_study/10_ollama_deploy.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OLLAMA_URL = "http://localhost:11434"


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _server_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _generate(model: str, prompt: str) -> dict:
    """Call Ollama /api/generate; return text + tokens/sec measured from the server's timings."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.3, "num_predict": 64}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    toks = d.get("eval_count", 0)
    secs = d.get("eval_duration", 0) / 1e9
    # tok/s is only meaningful with enough generated tokens; guard the degenerate (trial) case.
    tps = toks / secs if (secs > 0 and toks >= 8) else None
    return {"text": d.get("response", "").strip(), "tok_per_s": tps, "tokens": toks}


def run() -> dict:
    config.set_all_seeds()
    print(f"=== §10 Ollama deploy | mode={config.RUN_MODE} ===")
    if not shutil.which("ollama"):
        print("  [skip] no `ollama` binary. Install from https://ollama.com, then re-run.")
        return {"skipped": "no ollama"}
    if not _server_up():
        print("  [skip] Ollama server not reachable on :11434. Start it with `ollama serve` and re-run.")
        return {"skipped": "server down"}

    merged = config.OUTPUTS / f"merged_{config.RUN_MODE}"
    if not (merged / "Modelfile").exists():
        print("  (merged model/Modelfile missing — building via §9)")
        _sib("09_merge_and_gguf.py", "s9_for_10").run()

    name = f"chem-sft-{config.MODEL_KEY}-{config.RUN_MODE}"
    # Quantize to Q4_K_M on import for any 4-bit model — works whether the source is an F16 GGUF
    # (SmolLM3) or merged safetensors (MiniCPM5, Llama-arch → Ollama imports it directly). <2 GB.
    quant_flag = ["-q", "q4_K_M"] if config.LOAD_IN_4BIT else []
    existing = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if name in existing:
        print(f"  [cached] model '{name}' already imported.")
    else:
        how = "quantizing F16->Q4_K_M" if quant_flag else "importing"
        print(f"  {how} '{name}' via `ollama create`...")
        r = subprocess.run(["ollama", "create", name, *quant_flag, "-f", "Modelfile"], cwd=str(merged),
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [skip] ollama create failed:\n{r.stderr[-800:]}")
            print("  (SmolLM3 safetensors aren't importable by Ollama — produce a GGUF in §9 first.)")
            return {"skipped": "create failed"}
        print(f"  imported '{name}'.")

    print("\n--- querying the deployed model (local Ollama API) ---")
    results, tps = [], []
    for q in config.SAMPLE_QUESTIONS[: config.limits()["gen_samples"]]:
        out = _generate(name, q)
        if out["tok_per_s"] is not None:
            tps.append(out["tok_per_s"])
        results.append({"q": q, **out})
        rate = f"{out['tok_per_s']:.1f} tok/s" if out["tok_per_s"] is not None else "tok/s N/A"
        print(f"\nQ: {q}\nA: {out['text'] or '(empty)'}\n   ({rate}, {out['tokens']} tokens)")
    avg = sum(tps) / len(tps) if tps else None
    if avg is not None:
        print(f"\n  avg local inference: {avg:.1f} tok/s on this machine ({config.record_env()['cuda']})")
    else:
        print("\n  [tok/s N/A] the deployed model produced too few tokens to time — expected in TRIAL")
        print("  (the SFT'd 135M is barely trained and emits EOS early). Re-run after a FULL fine-tune.")
    print("  The DEPLOYMENT itself worked: Ollama imported the model and served it over its API.")
    print("  Next: §11 edge benchmark (same tok/s number, taken on a Raspberry Pi-class device).")
    out_path = config.OUTPUTS / f"ollama_{config.RUN_MODE}.json"
    out_path.write_text(json.dumps({"model": name, "avg_tok_per_s": avg, "deployed": True,
                                    "results": results}, indent=2))
    return {"model": name, "avg_tok_per_s": avg, "deployed": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy the fine-tuned model with Ollama and query it.")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run()


if __name__ == "__main__":
    main()
