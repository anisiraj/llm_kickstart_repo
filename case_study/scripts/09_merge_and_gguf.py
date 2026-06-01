r"""
09_merge_and_gguf.py — Part B step 1: merge the LoRA adapter, then export to GGUF.

WHY: a small model's payoff is on-device inference, and the on-device runtimes (llama.cpp, Ollama,
LM Studio) consume GGUF. The pipeline is: merge LoRA into the base → export F16 GGUF → quantize
(Q4_K_M). convert_hf_to_gguf.py does NOT accept PEFT adapters, so the merge MUST come first.

WHAT RUNS HERE vs WHAT NEEDS TOOLING
------------------------------------
- MERGE: always runs (PEFT merge_and_unload → a standalone safetensors model). We merge the §5 SFT
  adapter (it doesn't touch embeddings, so the merge is clean; a CPT adapter that adapts tied
  embed_tokens/lm_head needs more care — see PITFALLS / §3 gotcha).
- GGUF: needs llama.cpp. If `convert_hf_to_gguf.py` + `llama-quantize` are found (LLAMA_CPP env var or
  PATH), we run them; otherwise we print the exact recipe and emit a Modelfile so §10/Ollama can import
  the merged model directly (Ollama converts Llama-arch safetensors itself).

CONTRACT: TRIAL/FULL, self-sufficient (trains the §5 adapter if missing), idempotent, reproducible.

Run:  python case_study/09_merge_and_gguf.py
      LLAMA_CPP=/path/to/llama.cpp python case_study/09_merge_and_gguf.py   # to actually emit GGUF
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def merge_adapter() -> Path:
    """Merge the §5 SFT adapter into the base instruct model → standalone safetensors model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged = config.OUTPUTS / f"merged_{config.RUN_MODE}"
    if (merged / "config.json").exists():
        print(f"  [cached] merged model at {merged} ({_dir_size_mb(merged):.0f} MB)")
        return merged

    adapter = config.OUTPUTS / f"sft_instruct_{config.RUN_MODE}" / "adapter"
    if not adapter.exists():
        print("  (no SFT adapter — training it via §5)")
        _sib("05_sft.py", "s5_for_09").run_sft(init="instruct")

    print(f"  merging {adapter} into {config.INSTRUCT_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(config.INSTRUCT_MODEL)
    base = AutoModelForCausalLM.from_pretrained(config.INSTRUCT_MODEL, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    print(f"  merged model saved to {merged} ({_dir_size_mb(merged):.0f} MB)")
    return merged


def write_modelfile(merged: Path, gguf: Path | None) -> Path:
    """Emit an Ollama Modelfile for §10 (uses the GGUF if present, else the merged dir directly)."""
    src = f"./{gguf.name}" if gguf else "."
    mf = merged / "Modelfile"
    mf.write_text(
        f"FROM {src}\n"
        'PARAMETER temperature 0.3\n'
        'PARAMETER num_ctx 2048\n'
        'PARAMETER stop "<|im_end|>"\n'
        'SYSTEM "You are a concise assistant for computational and quantum chemistry."\n')
    return mf


def to_gguf(merged: Path) -> Path | None:
    """Convert→quantize with llama.cpp if available; otherwise print the recipe and skip."""
    llama = os.environ.get("LLAMA_CPP")
    convert = None
    if llama and (Path(llama) / "convert_hf_to_gguf.py").exists():
        convert = Path(llama) / "convert_hf_to_gguf.py"
    elif shutil.which("convert_hf_to_gguf.py"):
        convert = Path(shutil.which("convert_hf_to_gguf.py"))
    quantize = shutil.which("llama-quantize") or (Path(llama) / "llama-quantize" if llama else None)

    f16 = merged / "model-f16.gguf"
    quant = merged / f"model-{config.QUANT.lower()}.gguf"
    if convert:
        # Convert to F16 GGUF (no build needed). The small Q4_K_M (<2GB) is produced by Ollama's
        # `create -q q4_K_M` in §10 (no compiled llama-quantize required). If a built llama-quantize
        # IS present we make the Q4_K_M here directly.
        if not f16.exists():
            print(f"  converting merged model -> F16 GGUF via {convert} (no build needed) ...")
            subprocess.run([sys.executable, str(convert), str(merged), "--outfile", str(f16),
                            "--outtype", "f16"], check=True)
        print(f"  F16 GGUF: {f16} ({f16.stat().st_size/1e6:.0f} MB)")
        if quantize and Path(str(quantize)).exists():
            print(f"  quantizing -> {config.QUANT} (<2GB) ...")
            subprocess.run([str(quantize), str(f16), str(quant), config.QUANT], check=True)
            print(f"  {config.QUANT} GGUF: {quant} ({quant.stat().st_size/1e6:.0f} MB)")
            return quant
        print(f"  (Ollama will quantize F16 -> {config.QUANT} at import in §10, ~<2GB)")
        return f16

    print("\n  [skip GGUF] llama.cpp convert script not found. SmolLM3 can't be imported as safetensors")
    print("  by Ollama (unsupported arch), so for SmolLM3 you NEED a GGUF. One-time setup (no build):")
    print("    git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp")
    print("    ../.venv/bin/pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    print("  then re-run with:  LLAMA_CPP=~/llama.cpp bash case_study/run.sh smollm3 09 10 11 12 13")
    print("  (the 135M is LlamaForCausalLM, which Ollama imports directly — no GGUF needed there.)")
    return None


def run() -> dict:
    config.set_all_seeds()
    print(f"=== §9 merge + GGUF | mode={config.RUN_MODE} ===")
    merged = merge_adapter()
    gguf = to_gguf(merged)
    mf = write_modelfile(merged, gguf)
    print(f"\n  Modelfile written: {mf}")
    print("  Next: §10 deploy with Ollama (uses the GGUF if present, else imports the merged model).")
    return {"merged": str(merged), "gguf": str(gguf) if gguf else None, "modelfile": str(mf)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF.")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run()


if __name__ == "__main__":
    main()
