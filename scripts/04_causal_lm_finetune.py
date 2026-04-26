"""
04_causal_lm_finetune.py
------------------------
Causal LM fine-tuning (continued pre-training style) with a tiny GPT-2 model.
Uses DataCollatorForLanguageModeling so labels are auto-set.
Run with: python scripts/04_causal_lm_finetune.py
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import tempfile

MODEL = "gpt2"   # 117M params, no GPU needed for a toy run

# ── 1. Tokenizer ─────────────────────────────────────────────────────────────
print("=== 1. Load tokenizer + model ===")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token      # GPT-2 has no pad token by default

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tok.eos_token_id
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

# ── 2. Toy dataset ────────────────────────────────────────────────────────────
print("\n=== 2. Build toy dataset ===")
corpus = [
    "The transformer architecture revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "LoRA reduces trainable parameters by injecting low-rank matrices.",
    "Supervised fine-tuning teaches a model to follow instructions.",
    "Reinforcement learning from human feedback aligns models with human values.",
    "Quantization reduces model size by representing weights in lower precision.",
    "The key-value cache speeds up autoregressive generation significantly.",
] * 8   # repeat for enough training steps

raw_ds = Dataset.from_dict({"text": corpus})
splits  = raw_ds.train_test_split(test_size=0.2, seed=42)

# ── 3. Tokenise ───────────────────────────────────────────────────────────────
print("\n=== 3. Tokenise ===")
BLOCK = 64   # use short blocks for speed in this demo

def tokenize(batch):
    return tok(
        batch["text"],
        truncation=True,
        max_length=BLOCK,
        padding="max_length",
    )

tok_ds = splits.map(tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")
print("  train rows:", len(tok_ds["train"]))

# ── 4. DataCollatorForLanguageModeling ────────────────────────────────────────
# mlm=False → causal LM; auto sets labels = input_ids (with -100 on pad)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# ── 5. TrainingArguments ──────────────────────────────────────────────────────
print("\n=== 4. TrainingArguments ===")
with tempfile.TemporaryDirectory() as tmp:
    args = TrainingArguments(
        output_dir=tmp,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="no",
        weight_decay=0.01,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # ── 6. Trainer ───────────────────────────────────────────────────────────
    print("\n=== 5. Trainer ===")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        processing_class=tok,
        data_collator=collator,
    )
    result = trainer.train()
    print(f"  final loss: {result.training_loss:.4f}")
    metrics = trainer.evaluate()
    print(f"  eval  loss: {metrics['eval_loss']:.4f}")

# ── 7. Generation ─────────────────────────────────────────────────────────────
print("\n=== 6. Text generation ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)
prompt = "The transformer architecture"
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
    )
print(" ", tok.decode(out[0], skip_special_tokens=True))

print("\nCausal LM fine-tuning demo OK!")
