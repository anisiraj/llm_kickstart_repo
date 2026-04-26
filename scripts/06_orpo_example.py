"""
06_orpo_example.py
------------------
ORPO (Odds Ratio Preference Optimization) training with TRL.
Uses GPT-2 + tiny synthetic preference dataset so it runs on CPU.

NOTE: ORPO does NOT need a separate reference model — the log-odds ratio is
computed from the policy model itself.

Correct import path (confirmed TRL docs):
  from trl import ORPOTrainer, ORPOConfig   (available trl >= 0.9)

Run with: python scripts/06_orpo_example.py
Requirements: trl>=0.9, transformers, datasets, accelerate
"""

import torch
import tempfile
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Import ────────────────────────────────────────────────────────────────────
try:
    from trl import ORPOTrainer, ORPOConfig
    print("Imported ORPOTrainer, ORPOConfig from trl (>=0.9)")
except ImportError:
    # Older TRL: trl.experimental
    from trl.experimental.orpo import ORPOTrainer, ORPOConfig  # type: ignore
    print("Imported ORPOTrainer, ORPOConfig from trl.experimental.orpo")

# ── 1. Model + Tokenizer ─────────────────────────────────────────────────────
print("\n=== 1. Load model ===")
MODEL = "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tok.eos_token_id

# ── 2. Preference Dataset ────────────────────────────────────────────────────
# ORPOTrainer expects: "prompt", "chosen", "rejected" (plain strings)
print("\n=== 2. Build preference dataset ===")
data = {
    "prompt": [
        "What is LoRA?",
        "What is quantization?",
        "Explain fine-tuning.",
        "What is RLHF?",
        "What is a chat template?",
        "What is PEFT?",
    ] * 4,
    "chosen": [
        "LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model weights, reducing memory and compute requirements.",
        "Quantization compresses model weights to lower bit-widths (e.g., 4-bit, 8-bit), reducing memory usage with minimal accuracy loss.",
        "Fine-tuning continues training a pre-trained model on a smaller, task-specific dataset to adapt it for a particular use case.",
        "RLHF (Reinforcement Learning from Human Feedback) trains models to align with human preferences using a reward model.",
        "A chat template structures conversation turns (system, user, assistant) into a format the model was trained to understand.",
        "PEFT (Parameter-Efficient Fine-Tuning) adapts large models by training only a small subset of parameters.",
    ] * 4,
    "rejected": [
        "LoRA is a method to make models faster.",
        "Quantization removes parts of the model.",
        "Fine-tuning is training from scratch.",
        "RLHF uses robots to train models.",
        "A chat template is just a prompt.",
        "PEFT makes models smaller.",
    ] * 4,
}
ds = Dataset.from_dict(data)
splits = ds.train_test_split(test_size=0.2, seed=42)
print(f"  train={len(splits['train'])}  eval={len(splits['test'])}")

# ── 3. ORPOConfig ─────────────────────────────────────────────────────────────
print("\n=== 3. ORPOConfig + ORPOTrainer ===")
with tempfile.TemporaryDirectory() as tmp:
    config = ORPOConfig(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=8e-6,
        beta=0.1,               # λ: odds-ratio coefficient
        max_length=128,         # max_prompt_length removed in trl 1.x
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # ── 4. ORPOTrainer ────────────────────────────────────────────────────────
    # No ref_model needed — ORPO folds SFT + preference into one loss
    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tok,
    )

    print("  starting training …")
    result = trainer.train()
    print(f"  loss: {result.training_loss:.4f}")

print("\nORPO example OK!")
