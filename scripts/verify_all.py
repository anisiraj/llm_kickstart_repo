"""
verify_all.py
-------------
Quick verification that all key packages and APIs are importable and functional.

TWO VENVS are used in this repo (Unsloth pins trl<0.9, incompatible with DPO/GRPO):

  .venv     → Unsloth + SFT           trl 0.24  | scripts 01-04, 08
  .venv-rl  → DPO / ORPO / GRPO       trl 1.2   | scripts 05-07

Expected results:
  .venv/bin/python    scripts/verify_all.py  → 23/26  (DPO/ORPO/GRPO fail — expected)
  .venv-rl/bin/python scripts/verify_all.py  → 24/26  (sagemaker/unsloth absent — expected)

Run with:
  .venv/bin/python scripts/verify_all.py
  .venv-rl/bin/python scripts/verify_all.py
"""

import sys
import importlib

PASS = "  ✓"
FAIL = "  ✗"

results = []

def check(label, fn):
    try:
        fn()
        results.append((label, True, None))
        print(f"{PASS}  {label}")
    except Exception as e:
        results.append((label, False, str(e)))
        print(f"{FAIL}  {label}  — {e}")

# ── 1. Core packages ──────────────────────────────────────────────────────────
print("\n=== Core Packages ===")

def chk_torch():
    import torch
    assert torch.__version__, "no version"
    print(f"       torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
check("torch", chk_torch)

def chk_transformers():
    import transformers
    print(f"       transformers {transformers.__version__}", end="")
check("transformers", chk_transformers)

def chk_datasets():
    import datasets
    print(f"       datasets {datasets.__version__}", end="")
check("datasets", chk_datasets)

def chk_trl():
    import trl
    print(f"       trl {trl.__version__}", end="")
check("trl", chk_trl)

def chk_peft():
    import peft
    print(f"       peft {peft.__version__}", end="")
check("peft", chk_peft)

def chk_accelerate():
    import accelerate
    print(f"       accelerate {accelerate.__version__}", end="")
check("accelerate", chk_accelerate)

# ── 2. PyTorch basics ─────────────────────────────────────────────────────────
print("\n=== PyTorch ===")

def chk_tensor():
    import torch
    x = torch.tensor([1.0, 2.0])
    assert x.sum().item() == 3.0
check("tensor ops", chk_tensor)

def chk_autograd():
    import torch
    a = torch.tensor(2.0, requires_grad=True)
    b = a ** 2
    b.backward()
    assert a.grad.item() == 4.0
check("autograd", chk_autograd)

def chk_nn():
    import torch.nn as nn
    m = nn.Linear(4, 2)
    import torch
    out = m(torch.randn(3, 4))
    assert out.shape == (3, 2)
check("nn.Module", chk_nn)

def chk_dataloader():
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.randn(20, 4), torch.zeros(20).long())
    dl = DataLoader(ds, batch_size=5)
    xb, yb = next(iter(dl))
    assert xb.shape == (5, 4)
check("DataLoader", chk_dataloader)

def chk_clm_collator():
    from transformers import DataCollatorForLanguageModeling, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    import torch
    batch = collator([{"input_ids": [1, 2, 3, 4]}])
    assert "labels" in batch
check("DataCollatorForLanguageModeling(mlm=False)", chk_clm_collator)

# ── 3. HF Datasets ────────────────────────────────────────────────────────────
print("\n=== HF Datasets ===")

def chk_from_dict():
    from datasets import Dataset
    ds = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    assert len(ds) == 3
check("Dataset.from_dict", chk_from_dict)

def chk_map():
    from datasets import Dataset
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    ds2 = ds.map(lambda e: {"y": e["x"] * 2})
    assert ds2["y"] == [2, 4, 6]
check("Dataset.map", chk_map)

def chk_filter():
    from datasets import Dataset
    ds = Dataset.from_dict({"x": [1, 2, 3, 4]})
    filtered = ds.filter(lambda e: e["x"] > 2)
    assert len(filtered) == 2
check("Dataset.filter", chk_filter)

def chk_split():
    from datasets import Dataset
    ds = Dataset.from_dict({"x": list(range(20))})
    splits = ds.train_test_split(test_size=0.2, seed=42)
    assert "train" in splits and "test" in splits
check("train_test_split", chk_split)

# ── 4. HF Transformers ────────────────────────────────────────────────────────
print("\n=== HF Transformers ===")

def chk_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tok("hello world", return_tensors="pt")
    assert "input_ids" in enc
check("AutoTokenizer", chk_tokenizer)

def chk_training_args():
    from transformers import TrainingArguments
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            eval_strategy="epoch",    # ← not evaluation_strategy
            report_to="none",
        )
        assert args.eval_strategy == "epoch"
check("TrainingArguments(eval_strategy='epoch')", chk_training_args)

def chk_processing_class():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import tempfile
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = Dataset.from_dict({"input_ids": [[101, 102]], "attention_mask": [[1, 1]], "label": [0]})
    ds.set_format("torch")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    with tempfile.TemporaryDirectory() as tmp:
        t = Trainer(
            model=model,
            args=TrainingArguments(output_dir=tmp, report_to="none"),
            processing_class=tok,     # ← modern API
        )
    assert t is not None
check("Trainer(processing_class=tok)", chk_processing_class)

# ── 5. TRL trainers ───────────────────────────────────────────────────────────
print("\n=== TRL ===")

def chk_sfttrainer_import():
    from trl import SFTTrainer, SFTConfig
check("SFTTrainer, SFTConfig", chk_sfttrainer_import)

def chk_dpo_import():
    from trl import DPOTrainer, DPOConfig
check("DPOTrainer, DPOConfig", chk_dpo_import)

def chk_orpo_import():
    # ORPO lives in trl.experimental.orpo (confirmed up to trl 1.2.0)
    # NOT available via `from trl import ORPOTrainer` — that raises ImportError
    from trl.experimental.orpo import ORPOTrainer, ORPOConfig
check("ORPOTrainer, ORPOConfig (trl.experimental.orpo)", chk_orpo_import)

def chk_grpo_import():
    from trl import GRPOTrainer, GRPOConfig
check("GRPOTrainer, GRPOConfig", chk_grpo_import)

# ── 6. PEFT / LoRA ────────────────────────────────────────────────────────────
print("\n=== PEFT ===")

def chk_lora():
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["c_attn"], task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    print(f"       trainable={trainable:,} / total={total:,} ({trainable/total*100:.2f}%)", end="")
check("LoraConfig + get_peft_model", chk_lora)

# ── 7. AWS (import only) ──────────────────────────────────────────────────────
print("\n=== AWS (import checks only) ===")

def chk_boto3():
    import boto3, botocore
    print(f"       boto3={boto3.__version__}", end="")
check("boto3", chk_boto3)

def chk_sagemaker():
    # sagemaker 3.x restructured — sagemaker.huggingface is in 2.x only
    # For HF estimator use: pip install "sagemaker>=2.200,<3"
    import sagemaker
    from importlib.metadata import version
    ver = version("sagemaker")
    print(f"       sagemaker={ver} (HuggingFace estimator needs 2.x)", end="")
check("sagemaker SDK (core)", chk_sagemaker)

def chk_unsloth():
    from unsloth import FastLanguageModel
check("unsloth", chk_unsloth)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = [l for l, ok, _ in results if ok]
failed = [(l, e) for l, ok, e in results if not ok]
print(f"PASSED: {len(passed)}/{len(results)}")
if failed:
    print(f"FAILED: {len(failed)}")
    for label, err in failed:
        print(f"  ✗ {label}: {err}")
else:
    print("All checks passed!")
