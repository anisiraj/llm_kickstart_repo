r"""
05_sft.py — Supervised fine-tuning (SFT) on the small Q&A set, with COMPLETION-ONLY loss.

THE CONTRAST WITH §3
--------------------
§3 (CPT) used FULL causal loss — every token contributed (unmasked fraction ~100%). SFT is different:
the data is prompt→completion, and we compute loss on the **completion only** (the prompt is masked).
This script PRINTS the unmasked-token fraction so you can SEE it is well under 100% — that is the
prompt being masked (TRL `completion_only_loss=True`, the default for prompt-completion data). If you
forget this, the model learns to generate questions too and "echoes the prompt back" (see PITFALLS.md).

CONTRACT: TRIAL/FULL flag, self-sufficient (builds the seed Q&A if missing, downloads the model),
idempotent (caches outputs/sft_<init>_<mode>/), reproducible. Prints sample answers for visual check.

Run:  python case_study/05_sft.py                       # TRIAL, init=instruct
      python case_study/05_sft.py --init base            # SFT the base model instead
      CASE_STUDY_MODE=full python case_study/05_sft.py
"""
from __future__ import annotations
import argparse
import importlib.util
import json
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


def load_seed(n: int | None = None) -> list[dict]:
    """Return the prompt-completion Q&A, building it (via §2) if the JSONL is missing."""
    path = config.SFT_DIR / "seed_qa.jsonl"
    if not path.exists():
        _sib("02_data_availability.py", "da_for_05").build_sft_seed()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:n] if n else rows


def run_sft(init: str = "instruct", n_examples: int | None = None, force: bool = False,
            base_adapter: str | None = None, tag_extra: str = "") -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig

    config.set_all_seeds()
    lim = config.limits()
    model_name = config.INSTRUCT_MODEL if init == "instruct" else config.BASE_MODEL
    n = n_examples if n_examples is not None else (10 if config.is_trial() else None)
    tag = f"sft_{init}{tag_extra}_{config.RUN_MODE}"
    out_dir = config.OUTPUTS / tag

    if (out_dir / "metrics.json").exists() and not force:
        m = json.loads((out_dir / "metrics.json").read_text())
        print(f"[cached] {tag}: unmasked={m['unmasked_fraction']*100:.0f}%, {m['n_examples']} ex. --force to rerun.")
        return m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== §5 SFT [init={init}, completion-only loss] | mode={config.RUN_MODE} | device={device} ===")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_seed(n)
    from datasets import Dataset
    ds = Dataset.from_list(rows)   # prompt-completion shape -> completion_only_loss applies
    print(f"  {len(rows)} prompt-completion pairs (loss on the COMPLETION only)")

    args = SFTConfig(
        output_dir=str(out_dir / "ckpt"),
        per_device_train_batch_size=config.SFT["batch_size"],
        gradient_accumulation_steps=config.SFT["grad_accum"],
        num_train_epochs=config.SFT["epochs"], max_steps=lim["sft_max_steps"],
        learning_rate=config.SFT["lr"], max_length=config.SFT["max_seq_len"],
        completion_only_loss=True, packing=False,    # mask the prompt; no packing keeps masking clean
        bf16=(device == "cuda"), logging_steps=5, save_strategy="no", report_to=[], seed=config.SEED)
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32),
        args=args, train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(r=config.SFT["lora_r"], lora_alpha=config.SFT["lora_alpha"],
                               lora_dropout=config.SFT["lora_dropout"],
                               target_modules=config.SFT["target_modules"], task_type="CAUSAL_LM"))

    # PROVE the masking: pull one collated batch and count non -100 labels. Completion-only loss
    # masks the prompt tokens (-100), so the fraction is well under 100% (cf. ~100% for CPT in §3).
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    unmasked = (labels != -100).float().mean().item()
    print(f"  unmasked-token fraction = {unmasked*100:.0f}%  -> prompt is MASKED (completion-only loss)")

    trainer.train()
    model = trainer.model.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    # Visual check: feed the plain question (matches the prompt-completion training format).
    utils.generate_samples(model, tok, n=lim["gen_samples"], chat=False,
                           title=f"§5 SFT answers (init={init})")
    m = dict(tag=tag, init=init, model=model_name, n_examples=len(rows),
             unmasked_fraction=unmasked, completion_only_loss=True, max_steps=lim["sft_max_steps"],
             lr=config.SFT["lr"], env=config.record_env())
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\n  adapter + metrics saved to {out_dir}")
    print(f"  Unmasked {unmasked*100:.0f}% (vs ~100% for CPT in §3) = the completion-only mask is working.")
    print("  Next: §6 base-vs-instruct sweep over SFT-set size (the centerpiece experiment).")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT with completion-only loss on the seed Q&A.")
    ap.add_argument("--init", choices=["instruct", "base"], default="instruct")
    ap.add_argument("--n", type=int, help="number of Q&A examples (default: 10 trial / all full)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", choices=["trial", "full"])
    args = ap.parse_args()
    if args.mode:
        config.set_mode(args.mode)
    run_sft(init=args.init, n_examples=args.n, force=args.force)


if __name__ == "__main__":
    main()
