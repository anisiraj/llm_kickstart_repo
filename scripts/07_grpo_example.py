"""
07_grpo_example.py
------------------
GRPO (Group Relative Policy Optimization) training with TRL.
Uses a tiny in-memory math dataset (no internet needed).

Key fixes vs. naive handbook examples:
  • gsm8k has column "question", not "prompt" — we rename it
  • reward_funcs (plural) takes a list of callables
  • GRPOConfig.num_generations must be <= per_device_train_batch_size

Run with: python scripts/07_grpo_example.py
Requirements: trl>=0.9, transformers, datasets, accelerate
"""

import re
import torch
import tempfile
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# ── 1. Toy math dataset ───────────────────────────────────────────────────────
print("=== 1. Build toy math dataset ===")

# Mirrors gsm8k structure: "question" + "answer" columns
math_data = {
    "question": [
        "What is 2 + 2?",
        "What is 5 * 3?",
        "What is 10 - 4?",
        "What is 8 / 2?",
        "What is 3 ** 2?",
        "What is 12 + 7?",
        "What is 20 - 11?",
        "What is 6 * 6?",
    ] * 4,
    "answer": ["4", "15", "6", "4", "9", "19", "9", "36"] * 4,
}

ds = Dataset.from_dict(math_data)

# ── Key fix: rename "question" → "prompt" (GRPOTrainer expects "prompt") ──────
ds = ds.map(lambda x: {"prompt": x["question"]}, remove_columns=["question"])
print("  columns after rename:", ds.column_names)

splits = ds.train_test_split(test_size=0.2, seed=42)
print(f"  train={len(splits['train'])}  eval={len(splits['test'])}")

# ── 2. Model + Tokenizer ─────────────────────────────────────────────────────
print("\n=== 2. Load model ===")
MODEL = "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tok.eos_token_id

# ── 3. Reward functions ───────────────────────────────────────────────────────
# reward_funcs (plural) is a LIST of callables.
# Each function receives (prompts, completions, **kwargs) and returns List[float].
print("\n=== 3. Reward functions ===")

def reward_correct_answer(prompts, completions, **kwargs):
    """Give +1 if the completion contains the correct numeric answer."""
    answers = kwargs.get("answer", [""] * len(completions))
    rewards = []
    for completion, answer in zip(completions, answers):
        rewards.append(1.0 if answer in completion else 0.0)
    return rewards

def reward_brevity(prompts, completions, **kwargs):
    """Small bonus for short completions (encourage concise answers)."""
    return [max(0.0, 1.0 - len(c) / 200) for c in completions]

print("  defined reward_correct_answer + reward_brevity")

# ── 4. GRPOConfig ─────────────────────────────────────────────────────────────
print("\n=== 4. GRPOConfig ===")
with tempfile.TemporaryDirectory() as tmp:
    config = GRPOConfig(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        num_generations=4,             # must be <= per_device_train_batch_size
        max_completion_length=16,      # renamed from max_new_tokens in trl 1.x
        learning_rate=5e-6,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # ── 5. GRPOTrainer ────────────────────────────────────────────────────────
    print("\n=== 5. GRPOTrainer ===")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tok,
        reward_funcs=[reward_correct_answer, reward_brevity],   # ← list!
    )

    print("  starting training …")
    result = trainer.train()
    print(f"  loss: {result.training_loss:.4f}")

print("\nGRPO example OK!")
