"""
05_dpo_example.py
-----------------
DPO (Direct Preference Optimization) training with TRL.
Uses GPT-2 + tiny synthetic preference dataset so it runs on CPU.
Run with: python scripts/05_dpo_example.py

Requirements: trl>=0.9, transformers, datasets, peft, accelerate
"""

import torch
import tempfile
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# ── 1. Model + Tokenizer ─────────────────────────────────────────────────────
print("=== 1. Load model ===")
MODEL = "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tok.eos_token_id
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

# ── 2. Preference Dataset ────────────────────────────────────────────────────
# DPOTrainer expects: "prompt", "chosen", "rejected"  (plain strings)
print("\n=== 2. Build preference dataset ===")
data = {
    "prompt": [
        "What is machine learning?",
        "Explain gradient descent.",
        "What is overfitting?",
        "Define regularization.",
        "What is a neural network?",
        "What is backpropagation?",
    ] * 4,
    "chosen": [
        "Machine learning is a subset of AI where systems learn from data to improve performance.",
        "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent.",
        "Overfitting occurs when a model learns the training data too well, including noise, resulting in poor generalization.",
        "Regularization adds a penalty term to the loss function to reduce model complexity and prevent overfitting.",
        "A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes.",
        "Backpropagation computes gradients of the loss with respect to weights using the chain rule, enabling efficient training.",
    ] * 4,
    "rejected": [
        "Machine learning is when computers think like humans.",
        "Gradient descent goes downhill.",
        "Overfitting is when the model is too good.",
        "Regularization makes training harder.",
        "A neural network is a type of algorithm.",
        "Backpropagation sends signals backwards.",
    ] * 4,
}
ds = Dataset.from_dict(data)
splits = ds.train_test_split(test_size=0.2, seed=42)
print(f"  train={len(splits['train'])}  eval={len(splits['test'])}")
print("  columns:", ds.column_names)

# ── 3. DPOConfig ─────────────────────────────────────────────────────────────
print("\n=== 3. DPOConfig ===")
with tempfile.TemporaryDirectory() as tmp:
    config = DPOConfig(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        beta=0.1,               # KL penalty coefficient (temperature)
        max_length=128,         # max_prompt_length removed in trl 1.x
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # ── 4. DPOTrainer ─────────────────────────────────────────────────────────
    # ref_model=None → DPOTrainer uses the initial copy of model as reference
    print("\n=== 4. DPOTrainer ===")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # implicit reference model
        args=config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tok,
    )

    print("  starting training …")
    result = trainer.train()
    print(f"  loss: {result.training_loss:.4f}")
    metrics = trainer.evaluate()
    print(f"  eval loss: {metrics.get('eval_loss', 'n/a')}")

print("\nDPO example OK!")
