"""
10_evaluation.py
----------------
How to know if fine-tuning worked: loss tracking, perplexity, generation
quality checks, and before/after comparison.

Uses GPT-2 on a tiny corpus — no GPU required.

Run with: python scripts/10_evaluation.py
Requirements: transformers, datasets, torch
"""

import math
import torch
import tempfile
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL = "gpt2"

# ── 1. Setup ─────────────────────────────────────────────────────────────────
print("=== 1. Load model + tokenizer ===")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tok.eos_token_id
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

# ── 2. Dataset ────────────────────────────────────────────────────────────────
print("\n=== 2. Build dataset ===")
corpus = [
    "LoRA injects low-rank matrices into transformer attention layers.",
    "QLoRA combines 4-bit quantization with LoRA for memory efficiency.",
    "SFT teaches models to follow instructions from prompt-response pairs.",
    "DPO aligns models to preferences without a reward model.",
    "ORPO merges supervised fine-tuning and preference alignment into one loss.",
    "GRPO uses group-relative rewards for reinforcement learning.",
    "The learning rate is the most impactful hyperparameter in fine-tuning.",
    "Gradient accumulation lets you simulate larger batch sizes on small GPUs.",
] * 8

ds = Dataset.from_dict({"text": corpus})
splits = ds.train_test_split(test_size=0.2, seed=42)

BLOCK = 64

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=BLOCK, padding="max_length")

tok_ds = splits.map(tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")
print(f"  train={len(tok_ds['train'])}  eval={len(tok_ds['test'])}")

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# ── 3. Before-training baseline ──────────────────────────────────────────────
print("\n=== 3. Before-training baseline ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_sample(model, prompt, label=""):
    """Generate text from a prompt and print it."""
    model.eval()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"  [{label}] {text}")
    return text

def compute_perplexity(model, dataset, collator):
    """Compute perplexity over a dataset — the standard LM evaluation metric."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collator
    )

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            # Count non-padding tokens in labels
            labels = batch["labels"]
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# Baseline perplexity (before fine-tuning)
model.to(device)
baseline_loss, baseline_ppl = compute_perplexity(model, tok_ds["test"], collator)
print(f"  baseline eval loss: {baseline_loss:.4f}")
print(f"  baseline perplexity: {baseline_ppl:.2f}")

# Baseline generation
test_prompts = [
    "LoRA injects",
    "The learning rate",
    "GRPO uses",
]
print("\n  Before fine-tuning:")
for p in test_prompts:
    generate_sample(model, p, label="BEFORE")

# ── 4. Fine-tune ─────────────────────────────────────────────────────────────
print("\n=== 4. Fine-tune (2 epochs) ===")
with tempfile.TemporaryDirectory() as tmp:
    args = TrainingArguments(
        output_dir=tmp,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="no",
        weight_decay=0.01,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        processing_class=tok,
        data_collator=collator,
    )
    result = trainer.train()

    # ── 5. Inspect training history ──────────────────────────────────────────
    print("\n=== 5. Training history (trainer.state.log_history) ===")
    print("  This is the key object — it contains every logged metric.\n")

    train_losses = []
    eval_losses = []

    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            step = entry.get("step", "?")
            loss = entry["loss"]
            train_losses.append((step, loss))
            print(f"  step {step:>4d}  train_loss = {loss:.4f}")
        if "eval_loss" in entry:
            epoch = entry.get("epoch", "?")
            eloss = entry["eval_loss"]
            eval_losses.append((epoch, eloss))
            print(f"  epoch {epoch}     eval_loss = {eloss:.4f}")

    print(f"\n  Final training loss: {result.training_loss:.4f}")

    # ── 6. Check: is the loss actually decreasing? ───────────────────────────
    print("\n=== 6. Loss trend check ===")
    if len(train_losses) >= 2:
        first_loss = train_losses[0][1]
        last_loss = train_losses[-1][1]
        delta = first_loss - last_loss
        pct = (delta / first_loss) * 100 if first_loss > 0 else 0
        print(f"  first train_loss: {first_loss:.4f}")
        print(f"  last  train_loss: {last_loss:.4f}")
        print(f"  reduction: {delta:.4f} ({pct:.1f}%)")
        if delta > 0:
            print("  PASS — loss is decreasing")
        else:
            print("  WARN — loss is NOT decreasing; check lr, data, or epochs")

    if len(eval_losses) >= 2:
        first_eval = eval_losses[0][1]
        last_eval = eval_losses[-1][1]
        if last_eval > first_eval * 1.1:
            print("  WARN — eval loss increased: possible overfitting")
        else:
            print("  OK   — eval loss stable or decreasing")

# ── 7. After-training perplexity ─────────────────────────────────────────────
print("\n=== 7. After-training perplexity ===")
model.to(device)
after_loss, after_ppl = compute_perplexity(model, tok_ds["test"], collator)
print(f"  after eval loss:  {after_loss:.4f}  (was {baseline_loss:.4f})")
print(f"  after perplexity: {after_ppl:.2f}  (was {baseline_ppl:.2f})")
ppl_change = baseline_ppl - after_ppl
print(f"  perplexity dropped by {ppl_change:.2f}")
if after_ppl < baseline_ppl:
    print("  PASS — model improved on eval set")
else:
    print("  NOTE — perplexity did not improve (may need more data/epochs)")

# ── 8. After-training generation ─────────────────────────────────────────────
print("\n=== 8. Generation comparison (after fine-tuning) ===")
for p in test_prompts:
    generate_sample(model, p, label="AFTER")

# ── 9. Summary: your evaluation checklist ────────────────────────────────────
print("\n=== 9. Evaluation checklist ===")
print("""
  When fine-tuning your own model, always check:

  1. LOSS CURVE     — trainer.state.log_history
                      Train loss should decrease. Eval loss should not spike.

  2. PERPLEXITY     — math.exp(eval_loss)
                      Lower = model assigns higher probability to correct tokens.
                      Compare before vs after fine-tuning.

  3. GENERATION     — Generate from the same prompts before and after.
                      Does the output look more relevant? More coherent?

  4. OVERFITTING    — If eval_loss rises while train_loss drops, you're
                      memorizing the training set. Reduce epochs or add data.

  5. TASK-SPECIFIC  — For classification: accuracy, F1
                      For QA: exact match, ROUGE
                      For preference: win rate vs baseline
""")

print("Evaluation demo OK!")
