r"""
08_unsloth_vs_hf.py — head-to-head: the SAME SFT workload via HF/PEFT vs Unsloth.

This answers research question #5 (workflow differences) with measured numbers: wall-clock time,
peak VRAM, and trainable-parameter count, for an identical small SFT run. To keep it a clean
speed/VRAM comparison we train plain language-modeling on a "Q: .. / A: .." text field (loss
semantics are covered in §3/§5), with a FIXED step count so both backends do the same work.

THE ENV SPLIT (the workflow difference itself): Unsloth pins trl/xformers, so it lives in `.venv`;
the HF/PEFT path lives in `.venv-rl`. A single process can't import both. So this script measures the
backend available in the current interpreter, writes its numbers, then shells out to the OTHER venv's
python to measure the other, and finally prints the comparison.

CONTRACT: TRIAL/FULL, idempotent (caches outputs/bench_<backend>_<mode>.json), reproducible.

Run:  python case_study/08_unsloth_vs_hf.py            # measures HF here, shells to .venv for Unsloth
      ./.venv/bin/python case_study/08_unsloth_vs_hf.py --only unsloth   # (what the shell-out runs)
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parent              # case_study/scripts/
REPO = ROOT.parent.parent                            # repo root (holds .venv / .venv-rl)


def _sib(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _text_dataset():
    rows = _sib("05_sft.py", "s5_for_08").load_seed(None)
    from datasets import Dataset
    return Dataset.from_list([{"text": f"Q: {r['prompt']}\nA: {r['completion']}"} for r in rows])


def _bench_path(backend: str) -> Path:
    return config.OUTPUTS / f"bench_{backend}_{config.RUN_MODE}.json"


def measure_hf() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    lim = config.limits()
    tok = AutoTokenizer.from_pretrained(config.INSTRUCT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args = SFTConfig(output_dir="/tmp/bench_hf", per_device_train_batch_size=config.SFT["batch_size"],
                     gradient_accumulation_steps=1, max_steps=max(lim["sft_max_steps"], 10),
                     learning_rate=config.SFT["lr"], max_length=config.SFT["max_seq_len"],
                     dataset_text_field="text", packing=False, bf16=torch.cuda.is_available(),
                     logging_steps=1000, save_strategy="no", report_to=[], seed=config.SEED)
    model = AutoModelForCausalLM.from_pretrained(
        config.INSTRUCT_MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    trainer = SFTTrainer(model=model, args=args, train_dataset=_text_dataset(), processing_class=tok,
                         peft_config=LoraConfig(r=config.SFT["lora_r"], lora_alpha=config.SFT["lora_alpha"],
                                                target_modules=config.SFT["target_modules"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    t0 = time.perf_counter(); trainer.train(); dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return dict(backend="hf", seconds=dt, peak_vram_gb=vram, trainable_params=trainable,
                steps=args.max_steps, env=config.record_env())


def measure_unsloth() -> dict:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
    lim = config.limits()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.INSTRUCT_MODEL, max_seq_length=config.SFT["max_seq_len"],
        dtype=None, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=config.SFT["lora_r"], lora_alpha=config.SFT["lora_alpha"],
        target_modules=config.SFT["target_modules"], use_gradient_checkpointing="unsloth",
        random_state=config.SEED)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args = SFTConfig(output_dir="/tmp/bench_unsloth", per_device_train_batch_size=config.SFT["batch_size"],
                     gradient_accumulation_steps=1, max_steps=max(lim["sft_max_steps"], 10),
                     learning_rate=config.SFT["lr"], max_seq_length=config.SFT["max_seq_len"],
                     dataset_text_field="text", packing=False, bf16=is_bfloat16_supported(),
                     logging_steps=1000, save_strategy="no", report_to=[], seed=config.SEED)
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=_text_dataset(), args=args)
    t0 = time.perf_counter(); trainer.train(); dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return dict(backend="unsloth", seconds=dt, peak_vram_gb=vram, trainable_params=trainable,
                steps=args.max_steps, env=config.record_env())


def _current_backend() -> str:
    return "unsloth" if importlib.util.find_spec("unsloth") else "hf"


def _other_python(cur: str) -> Path:
    return REPO / (".venv/bin/python" if cur == "hf" else ".venv-rl/bin/python")


def run(force: bool = False) -> dict:
    config.set_all_seeds()
    cur = _current_backend()
    print(f"=== §8 Unsloth-vs-HF | mode={config.RUN_MODE} | this interpreter = {cur} ===")

    # measure the backend available here
    p = _bench_path(cur)
    if p.exists() and not force:
        print(f"  [cached] {cur}: {json.loads(p.read_text())['seconds']:.2f}s")
    else:
        m = (measure_unsloth if cur == "unsloth" else measure_hf)()
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(m, indent=2))
        print(f"  measured {cur}: {m['seconds']:.2f}s | peak VRAM {m['peak_vram_gb']:.2f} GB | "
              f"trainable {m['trainable_params']:,}")

    # shell out to the other env (only at top level, not recursively)
    other = "hf" if cur == "unsloth" else "unsloth"
    op = _bench_path(other)
    other_py = _other_python(cur)
    if not op.exists() or force:
        if other_py.exists():
            print(f"  shelling out to {other_py.name}-env to measure {other}...")
            subprocess.run([str(other_py), str(ROOT / "08_unsloth_vs_hf.py"), "--only", other,
                            "--mode", config.RUN_MODE] + (["--force"] if force else []),
                           cwd=str(ROOT))
        else:
            print(f"  [skip] {other} env not found at {other_py} — run it there manually.")

    _compare()
    return {"compared": [cur, other]}


def _compare() -> None:
    hf, us = _bench_path("hf"), _bench_path("unsloth")
    if not (hf.exists() and us.exists()):
        print("  (run in both envs to see the comparison table)")
        return
    a, b = json.loads(hf.read_text()), json.loads(us.read_text())
    print("\n=== HF vs Unsloth (same SFT workload, measured on this machine) ===")
    print(f"  {'metric':18} | {'HF/PEFT':>12} | {'Unsloth':>12}")
    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'wall time (s)':18} | {a['seconds']:>12.2f} | {b['seconds']:>12.2f}")
    print(f"  {'peak VRAM (GB)':18} | {a['peak_vram_gb']:>12.2f} | {b['peak_vram_gb']:>12.2f}")
    print(f"  {'trainable params':18} | {a['trainable_params']:>12,} | {b['trainable_params']:>12,}")
    if b["seconds"]:
        print(f"\n  Unsloth speedup vs HF: {a['seconds']/b['seconds']:.2f}x (this run, this GPU)")
    print("  Note: trainable-param differences reflect each backend's LoRA/embedding choices (§3 §9).")


def main() -> None:
    ap = argparse.ArgumentParser(description="HF vs Unsloth speed/VRAM head-to-head.")
    ap.add_argument("--only", choices=["hf", "unsloth"], help="measure just one backend (no shell-out)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    if args.only:
        fn = measure_unsloth if args.only == "unsloth" else measure_hf
        p = _bench_path(args.only)
        if p.exists() and not args.force:
            print(f"[cached] {args.only}")
            return
        m = fn(); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(m, indent=2))
        print(f"measured {args.only}: {m['seconds']:.2f}s | VRAM {m['peak_vram_gb']:.2f} GB")
    else:
        run(force=args.force)


if __name__ == "__main__":
    main()
