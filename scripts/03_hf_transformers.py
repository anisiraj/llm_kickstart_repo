"""
03_hf_transformers.py
---------------------
Working HuggingFace Transformers examples: tokenizer, model, inference, Trainer.
Uses tiny/distilled models so it runs on CPU without a GPU.
Run with: python scripts/03_hf_transformers.py
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
from datasets import Dataset

# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
print("=== 1. Tokenizer ===")
MODEL = "distilbert-base-uncased"   # tiny model, ~66M params, no GPU needed
tok = AutoTokenizer.from_pretrained(MODEL)

text = "HuggingFace Transformers are great!"
enc = tok(text, return_tensors="pt")
print("input_ids shape:", enc["input_ids"].shape)
print("decoded:", tok.decode(enc["input_ids"][0]))

# Batch encode with padding/truncation
batch = tok(
    ["Short sentence.", "A much longer sentence that will need padding."],
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)
print("batch input_ids shape:", batch["input_ids"].shape)

# ── 2. AutoModel for inference ────────────────────────────────────────────────
print("\n=== 2. AutoModel (sequence classification) ===")
clf = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
clf.eval()
with torch.no_grad():
    outputs = clf(**enc)
print("logits:", outputs.logits)
print("predicted class:", outputs.logits.argmax(-1).item())

# ── 3. pipeline API ───────────────────────────────────────────────────────────
print("\n=== 3. pipeline API ===")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
results = sentiment(["I love this!", "This is terrible."])
for r in results:
    print(f"  {r['label']:8s}  score={r['score']:.3f}")

# ── 4. TrainingArguments (no actual training – just validates API) ────────────
print("\n=== 4. TrainingArguments ===")
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    args = TrainingArguments(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        eval_strategy="epoch",      # ← correct (not evaluation_strategy, deprecated v4.38)
        save_strategy="epoch",
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
    )
    print("  output_dir:", args.output_dir)
    print("  eval_strategy:", args.eval_strategy)
    print("  learning_rate:", args.learning_rate)

# ── 5. Trainer (tiny dataset, classification) ─────────────────────────────────
print("\n=== 5. Trainer (tiny run) ===")

texts  = ["I love NLP!", "I hate bugs.", "Models are fun.", "Training is slow."] * 4
labels = [1, 0, 1, 0] * 4

raw_ds = Dataset.from_dict({"text": texts, "label": labels})
splits  = raw_ds.train_test_split(test_size=0.25, seed=42)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=32)

tok_ds = splits.map(tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")
collator = DataCollatorWithPadding(tok)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

with tempfile.TemporaryDirectory() as tmp:
    t_args = TrainingArguments(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        processing_class=tok,          # ← modern API (replaces tokenizer= kwarg)
        data_collator=collator,
    )
    train_result = trainer.train()
    print("  train loss:", round(train_result.training_loss, 4))
    metrics = trainer.evaluate()
    print("  eval loss:", round(metrics["eval_loss"], 4))

# ── 6. Save + reload ─────────────────────────────────────────────────────────
print("\n=== 6. Save + reload ===")
with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    m2 = AutoModelForSequenceClassification.from_pretrained(tmp)
    t2 = AutoTokenizer.from_pretrained(tmp)
    print("  reloaded vocab size:", t2.vocab_size)

print("\nAll HF Transformers basics OK!")
