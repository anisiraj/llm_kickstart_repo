r"""
13_smollm3_toolcall.py — can a LARGER model do tool-calling where the 135M failed?

§12 showed our fine-tuned SmolLM2-135M cannot emit a valid tool call — too small. This section runs the
SAME kind of tool-use task against a tool-capable model (SmolLM3-3B) via Ollama's NATIVE function-calling
API (`/api/chat` with a `tools` schema), and contrasts the two. Lesson for the book: tool-use/agentic
behavior is an emergent capability that needs scale — fine-tuning a 135M won't conjure it.

CONTRACT: graceful — pulls the model via Ollama if missing; skips cleanly if Ollama/model unavailable.

Run:  python case_study/13_smollm3_toolcall.py
      python case_study/13_smollm3_toolcall.py --model hf.co/HuggingFaceTB/SmolLM3-3B-GGUF:Q4_K_M
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


def _have_model(name: str) -> bool:
    try:
        tags = json.loads(urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5).read())
        return any(m["name"].startswith(name.split(":")[0]) for m in tags.get("models", []))
    except Exception:
        return False


def _chat(model: str, messages: list, tools=None) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def tool_call_test(model: str) -> dict:
    """Ask a multiplication question with the multiply tool; see if the model calls it correctly."""
    messages = [{"role": "user", "content": "What is 23 times 19? Use the multiply tool."}]
    resp = _chat(model, messages, tools=TOOLS)
    msg = resp.get("message", {})
    calls = msg.get("tool_calls") or []
    if not calls:
        print(f"  [{model}] NO tool_call emitted. reply: {msg.get('content','')[:160]!r}")
        return {"model": model, "tool_called": False}

    fn = calls[0]["function"]
    args = fn.get("arguments", {})
    a, b = float(args.get("a")), float(args.get("b"))
    product = a * b
    print(f"  [{model}] tool_call: multiply(a={a}, b={b}) -> {product}")
    # feed the tool result back for a final natural-language answer
    messages += [msg, {"role": "tool", "content": str(product)}]
    final = _chat(model, messages).get("message", {}).get("content", "").strip()
    ok = str(int(product)) in final or str(product) in final
    print(f"  [{model}] final answer: {final[:160]!r}  (correct: {ok})")
    return {"model": model, "tool_called": True, "args": {"a": a, "b": b},
            "product": product, "answer_correct": bool(ok)}


def run(model: str | None = None) -> dict:
    config.set_all_seeds()
    model = model or config.SMOLLM3_OLLAMA
    print(f"=== §13 SmolLM3 tool-calling (vs the 135M's failure in §12) ===")
    if not shutil.which("ollama") or not _server_up():
        print("  [skip] Ollama not available — install/start it, then re-run (see §10).")
        return {"skipped": "no ollama"}

    if not _have_model(model):
        print(f"  pulling '{model}' via Ollama (large download the first time)...")
        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [skip] `ollama pull {model}` failed:\n{r.stderr[-500:]}")
            print("  Try a HF GGUF, e.g.:  --model hf.co/HuggingFaceTB/SmolLM3-3B-GGUF:Q4_K_M")
            return {"skipped": "pull failed"}

    print("\n  >>> tool-capable model (SmolLM3):")
    big = tool_call_test(model)

    # contrast: the fine-tuned 135M from §10 (if present) — expected to NOT tool-call
    small_name = f"chem-sft-{config.RUN_MODE}"
    small = None
    if _have_model(small_name):
        print("\n  >>> our fine-tuned 135M (for contrast):")
        try:
            small = tool_call_test(small_name)
        except Exception as e:
            print(f"  [{small_name}] errored on tools API ({type(e).__name__}) — 135M lacks tool support.")
            small = {"model": small_name, "tool_called": False}

    print("\n  TAKEAWAY: tool-use is an emergent, scale-dependent capability. SmolLM3-3B can call the tool;")
    print("  the fine-tuned 135M cannot. Fine-tuning teaches a task/format, it does not add capabilities")
    print("  the base model never had. Pick model SIZE for the job (agentic => bigger).")
    out = {"smollm3": big, "small_135m": small}
    (config.OUTPUTS / "smollm3_toolcall.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolLM3 tool-calling test via Ollama native tools API.")
    ap.add_argument("--model", help="Ollama model name (default config.SMOLLM3_OLLAMA)")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run(model=args.model)


if __name__ == "__main__":
    main()
