r"""
12_harness.py — Part B step 4: run the model through a HARNESS (agent loop + eval harness).

Two senses of "harness":
  (A) AGENT harness — drive the deployed model in a minimal tool-use loop: offer it a tool, parse a
      tool call from its reply, execute it, feed the result back. Small models are WEAK at this, so we
      report honestly whether the 135M managed it — the point is the harness mechanics, not a win.
  (B) EVAL harness — EleutherAI's lm-evaluation-harness (`lm-eval`) is the standard for benchmarking.
      It isn't installed here, so we print the exact command to evaluate this model and skip gracefully.

CONTRACT: self-sufficient (deploys via §10 if needed), graceful skips, reproducible.

Run:  python case_study/12_harness.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import shutil
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


# ── (A) Minimal agent harness ─────────────────────────────────────────────────
def _tool_multiply(a: float, b: float) -> float:
    return a * b


def _chat(name: str, prompt: str, n_predict: int = 80) -> str:
    body = json.dumps({"model": name, "prompt": prompt, "stream": False,
                       "options": {"num_predict": n_predict, "temperature": 0.0}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("response", "")


def agent_demo(name: str) -> dict:
    """One-step tool-use loop: the model is told to emit MULTIPLY(a,b); we execute and feed back."""
    question = "What is 23 times 19? Use the tool."
    sys_prompt = (
        "You can call one tool by writing exactly MULTIPLY(a, b). "
        f"User: {question}\nAssistant:")
    reply = _chat(name, sys_prompt)
    call = re.search(r"MULTIPLY\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", reply)
    print(f"\n  [agent] model reply: {reply.strip()[:160]!r}")
    if call:
        a, b = float(call.group(1)), float(call.group(2))
        result = _tool_multiply(a, b)
        final = _chat(name, sys_prompt + reply + f"\nTOOL_RESULT={result}\nAssistant:")
        ok = str(int(result)) in final or str(result) in final
        print(f"  [agent] parsed MULTIPLY({a},{b}) -> tool returned {result}; model final: {final.strip()[:120]!r}")
        print(f"  [agent] tool-use {'succeeded' if call else 'attempted'}; correct answer surfaced: {ok}")
        return {"tool_called": True, "result": result, "answer_correct": bool(ok)}
    print("  [agent] model did NOT emit a valid tool call — expected for a 135M model. The harness")
    print("          mechanics (offer tool -> parse -> execute -> feed back) are what this demonstrates.")
    return {"tool_called": False}


# ── (B) lm-evaluation-harness recipe ──────────────────────────────────────────
def eval_harness_recipe() -> dict:
    if importlib.util.find_spec("lm_eval") is None:
        merged = config.OUTPUTS / f"merged_{config.RUN_MODE}"
        print("\n  [skip lm-eval] not installed. To benchmark this model with the standard harness:")
        print("    pip install lm-eval")
        print(f"    lm_eval --model hf --model_args pretrained={merged} \\")
        print("            --tasks hellaswag,arc_easy --device cuda:0 --batch_size 8")
        print("    (always pass an explicit tokenizer for GGUF models; reconstructing it can hang).")
        return {"lm_eval": "not installed (recipe printed)"}
    print("  lm-eval is installed — run the command above to produce a benchmark scorecard.")
    return {"lm_eval": "installed"}


def run() -> dict:
    config.set_all_seeds()
    print(f"=== §12 harness (agent + eval) | mode={config.RUN_MODE} ===")
    out = {"agent": None, "eval": None}
    if not shutil.which("ollama"):
        print("  [skip agent] no `ollama` — see §10.")
    else:
        name = f"chem-sft-{config.RUN_MODE}"
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
            tags = json.loads(urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3).read())
            if not any(mm["name"].startswith(name) for mm in tags.get("models", [])):
                _sib("10_ollama_deploy.py", "s10_for_12").run()
            out["agent"] = agent_demo(name)
        except Exception as e:
            print(f"  [skip agent] Ollama not reachable: {type(e).__name__}")
    out["eval"] = eval_harness_recipe()
    print("\n  Part B complete: train -> merge -> GGUF/Ollama -> edge benchmark -> harness.")
    (config.OUTPUTS / f"harness_{config.RUN_MODE}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent loop + lm-evaluation-harness recipe.")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run()


if __name__ == "__main__":
    main()
