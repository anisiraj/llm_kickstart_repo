r"""
13_smollm3_toolcall.py — can OUR fine-tuned model do tool-calling? (size vs capability)

§12 showed our fine-tuned SmolLM2-135M cannot emit a valid tool call — too small. The interesting
question is whether the SAME recipe on a bigger model (our fine-tuned SmolLM3-3B) CAN. This tests
**our own deployed fine-tunes** through Ollama's native function-calling API (`/api/chat` + `tools`):
  • chem-sft-smollm3-<mode>      — our QLoRA-fine-tuned SmolLM3-3B   (expected: tool-calls)
  • chem-sft-smollm2-135m-<mode> — our fine-tuned 135M               (expected: cannot)
Lesson: tool-use is an emergent, scale-dependent capability — fine-tuning teaches a task/format, it
does not add a capability the base model never had. Pick model SIZE for the job.

OPTIONAL CONTROL (`--stock`): also pull stock SmolLM3 and test it, to confirm our domain fine-tuning
did not *break* tool-calling. Off by default — the point is our models, no extra download needed.

CONTRACT: graceful — uses whatever fine-tuned models are already deployed (via the §10 step of each
pipeline); skips cleanly if Ollama/model unavailable.

Run:  python case_study/13_smollm3_toolcall.py
      python case_study/13_smollm3_toolcall.py --stock     # also test stock SmolLM3 as a control
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OLLAMA_URL = "http://localhost:11434"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers and return the product.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]},
    },
}]


def _server_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3); return True
    except Exception:
        return False


def _deployed() -> set[str]:
    try:
        tags = json.loads(urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5).read())
        return {m["name"] for m in tags.get("models", [])}
    except Exception:
        return set()


def _has(name: str, deployed: set[str]) -> bool:
    return any(d.startswith(name) for d in deployed)


def _chat(model: str, messages: list, tools=None) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def tool_call_test(model: str) -> dict:
    """Ask a multiplication question with the multiply tool; did the model call it correctly?"""
    messages = [{"role": "user", "content": "What is 23 times 19? Use the multiply tool."}]
    try:
        resp = _chat(model, messages, tools=TOOLS)
    except Exception as e:
        print(f"  [{model}] tools API error ({type(e).__name__}) — model likely has no tool template.")
        return {"model": model, "tool_called": False, "error": type(e).__name__}
    msg = resp.get("message", {})
    calls = msg.get("tool_calls") or []
    if not calls:
        print(f"  [{model}] NO tool_call. reply: {msg.get('content','')[:140]!r}")
        return {"model": model, "tool_called": False}
    args = calls[0]["function"].get("arguments", {})
    a, b = float(args.get("a")), float(args.get("b"))
    product = a * b
    messages += [msg, {"role": "tool", "content": str(product)}]
    final = _chat(model, messages).get("message", {}).get("content", "").strip()
    ok = str(int(product)) in final or str(product) in final
    print(f"  [{model}] tool_call multiply({a},{b})={product}; final: {final[:120]!r} (correct={ok})")
    return {"model": model, "tool_called": True, "product": product, "answer_correct": bool(ok)}


def run(include_stock: bool = False) -> dict:
    config.set_all_seeds()
    mode = config.RUN_MODE
    print("=== §13 tool-calling: does OUR fine-tuned model do it? (size vs capability) ===")
    if not shutil.which("ollama") or not _server_up():
        print("  [skip] Ollama not available — deploy our models via the §10 step first (see §10).")
        return {"skipped": "no ollama"}

    deployed = _deployed()
    out = {}
    # our fine-tuned models (deployed by each pipeline's §10), big first
    for key in ("smollm3", "smollm2-135m"):
        name = f"chem-sft-{key}-{mode}"
        if _has(name, deployed):
            print(f"\n  >>> our fine-tuned {key}:")
            out[key] = tool_call_test(name)
        else:
            print(f"\n  [not deployed] {name} — run `run.sh {key}` (it deploys via §10) to include it.")

    if include_stock:
        m = config.SMOLLM3_OLLAMA
        if not _has(m, deployed):
            print(f"\n  pulling stock control '{m}'...")
            if subprocess.run(["ollama", "pull", m]).returncode != 0:
                print("  [skip stock] pull failed.")
                m = None
        if m:
            print("\n  >>> stock SmolLM3 (control — did our fine-tuning break tools?):")
            out["stock_smollm3"] = tool_call_test(m)

    print("\n  TAKEAWAY: tool-use is emergent/scale-dependent. If our 3B tool-calls and our 135M does not,")
    print("  that gap is about MODEL SIZE, not fine-tuning — the recipe is identical for both.")
    (config.OUTPUTS / "toolcall.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tool-calling test on our fine-tuned models (Ollama tools API).")
    ap.add_argument("--stock", action="store_true", help="also test stock SmolLM3 as a control")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run(include_stock=args.stock)


if __name__ == "__main__":
    main()
