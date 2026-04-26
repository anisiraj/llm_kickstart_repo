"""
12_dataset_ecosystem.py
-----------------------
The Dataset Zoo — PyTorch Dataset vs HF Dataset vs DataLoader vs DataCollator.

Senior devs hit this wall: four things called "dataset" and nobody explains how
they connect. This script walks through each layer, shows what it does, and
reveals what the Trainer hides from you.

Run with: python scripts/12_dataset_ecosystem.py
"""

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import tempfile

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: PyTorch's Native Dataset — The Foundation
# ═══════════════════════════════════════════════════════════════════════════════
# torch.utils.data.Dataset is an abstract class. You subclass it and implement
# __len__() and __getitem__(). This is how ALL data starts in PyTorch.
#
# Think of it as: "given an index, return one sample."
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A: PyTorch Dataset — The Foundation")
print("=" * 70)


class SentimentDataset(TorchDataset):
    """Classic PyTorch Dataset — you write the class, you control everything."""

    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",       # fixed-length — crude but simple
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

texts = [
    "This product is amazing, highly recommend!",
    "Terrible experience, waste of money.",
    "Not bad, does the job.",
    "Absolutely love this, will buy again!",
] * 10  # repeat for demo

labels = [1, 0, 1, 1] * 10

pt_dataset = SentimentDataset(texts, labels, tok)
print(f"  PyTorch Dataset length: {len(pt_dataset)}")
print(f"  Single item keys: {list(pt_dataset[0].keys())}")
print(f"  input_ids shape:  {pt_dataset[0]['input_ids'].shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: DataLoader — Batching, Shuffling, Parallelism
# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader wraps ANY Dataset and handles:
#   - Batching (stack multiple items into a batch)
#   - Shuffling (randomize order each epoch)
#   - num_workers (parallel data loading)
#   - collate_fn (HOW to combine items into a batch)
#
# The default collate_fn just stacks tensors. But what if sequences have
# different lengths? That's where collate_fn gets interesting.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B: DataLoader — Batching + Shuffling")
print("=" * 70)

loader = DataLoader(
    pt_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,      # 0 = main process (safe for demos)
    # collate_fn=None → default: torch.stack each field
)

batch = next(iter(loader))
print(f"  Batch keys:          {list(batch.keys())}")
print(f"  input_ids shape:     {batch['input_ids'].shape}")   # [4, 64]
print(f"  attention_mask shape: {batch['attention_mask'].shape}")
print(f"  labels shape:        {batch['labels'].shape}")

# ── The problem with fixed padding ───────────────────────────────────────────
# Above, we padded every sequence to max_length=64 inside __getitem__.
# That works but wastes compute — most sequences are shorter than 64.
# The fix: pad to the longest sequence IN THE BATCH, not globally.
# That's exactly what a DataCollator does.

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: HuggingFace Dataset — The Replacement You Didn't Know You Needed
# ═══════════════════════════════════════════════════════════════════════════════
# datasets.Dataset looks like a PyTorch Dataset but is completely different:
#   - Arrow-backed (memory-mapped, doesn't load everything into RAM)
#   - Column-oriented (like a DataFrame, not row-oriented)
#   - Has .map(), .filter(), .select() — transforms without copying
#   - Can convert to PyTorch format with .set_format("torch")
#
# KEY INSIGHT: HF Dataset IS compatible with DataLoader.
# After .set_format("torch"), it behaves like a PyTorch Dataset.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART C: HuggingFace Dataset — Column-Oriented + Arrow-Backed")
print("=" * 70)

# Create from dict (no subclassing needed!)
hf_ds = HFDataset.from_dict({
    "text": texts,
    "label": labels,
})
print(f"  HF Dataset: {hf_ds}")
print(f"  Columns:    {hf_ds.column_names}")
print(f"  First row:  {hf_ds[0]}")

# Tokenize with .map() — no __getitem__ override needed
def tokenize_fn(examples):
    return tok(examples["text"], truncation=True, max_length=64)
    # NOTE: no padding here! We'll let the collator handle that.

tok_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"\n  After tokenize .map():")
print(f"  Columns: {tok_ds.column_names}")
print(f"  Row 0 input_ids length: {len(tok_ds[0]['input_ids'])}")
print(f"  Row 1 input_ids length: {len(tok_ds[1]['input_ids'])}")
print(f"  ^ Different lengths! No padding yet — that's deliberate.")

# ── .set_format("torch") makes it act like a PyTorch Dataset ────────────────
tok_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print(f"\n  After set_format('torch'):")
print(f"  Type of input_ids: {type(tok_ds[0]['input_ids'])}")

# ── You CAN use it directly with DataLoader ─────────────────────────────────
# But you'll get an error if sequences have different lengths
# (default collate can't stack ragged tensors). That's why you need a collator.

# ═══════════════════════════════════════════════════════════════════════════════
# PART D: DataCollator — The Glue Between Dataset and Training
# ═══════════════════════════════════════════════════════════════════════════════
# A DataCollator is just a fancy collate_fn. It takes a list of individual
# examples and returns a batch dict of tensors.
#
# What collators typically do:
#   1. DYNAMIC PADDING — pad to longest in batch (not global max)
#   2. CREATE LABELS  — for language modelling, labels = shifted input_ids
#   3. MASK TOKENS    — for MLM, randomly replace tokens with [MASK]
#
# Different collators for different tasks:
#
#   DataCollatorWithPadding         → classification (just pads)
#   DataCollatorForLanguageModeling → causal LM (labels = input_ids, -100 on pad)
#   DataCollatorForSeq2Seq          → encoder-decoder (pads both input + target)
#   DataCollatorForCompletionOnlyLM → SFT (masks prompt tokens with -100,
#                                     only trains on completion)  [from trl]
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART D: DataCollator — Dynamic Padding + Label Creation")
print("=" * 70)

# ── D1: DataCollatorWithPadding (classification) ────────────────────────────
print("\n--- D1: DataCollatorWithPadding ---")
collator_pad = DataCollatorWithPadding(tokenizer=tok)

# Reset format to get plain lists (collator expects un-tensorified data)
tok_ds.reset_format()

# Take 3 examples with different lengths
samples = [tok_ds[i] for i in range(3)]
print(f"  Before collation:")
for i, s in enumerate(samples):
    print(f"    Sample {i}: {len(s['input_ids'])} tokens")

batch = collator_pad(samples)
print(f"  After collation:")
print(f"    input_ids shape: {batch['input_ids'].shape}")
print(f"    ^ All padded to {batch['input_ids'].shape[1]} (longest in batch)")

# ── D2: DataCollatorForLanguageModeling (causal LM) ─────────────────────────
print("\n--- D2: DataCollatorForLanguageModeling ---")
collator_lm = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
# mlm=False → causal LM: labels = input_ids, padding tokens get -100

# For LM, we need a dataset without the label column
lm_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
lm_samples = [lm_ds[i] for i in range(3)]

batch_lm = collator_lm(lm_samples)
print(f"  input_ids shape: {batch_lm['input_ids'].shape}")
print(f"  labels shape:    {batch_lm['labels'].shape}")
print(f"  labels[0][:5]:   {batch_lm['labels'][0][:5].tolist()}")
print(f"  ^ Labels = input_ids (model learns to predict next token)")

# Show that padding positions get -100 (ignored in loss)
pad_mask = batch_lm["labels"][0] == -100
print(f"  Positions with -100: {pad_mask.sum().item()} "
      f"(these are padding — excluded from loss)")

# ── D3: MLM collator (BERT-style only — GPT-2 can't do this) ────────────────
print("\n--- D3: DataCollatorForLanguageModeling (mlm=True) ---")
print("  mlm=True requires a [MASK] token (BERT, RoBERTa, etc.)")
print("  GPT-2 is causal-only, so we skip the live demo here.")
print("  For BERT: collator randomly masks ~15% of tokens,")
print("  sets labels = original token IDs, -100 on non-masked positions.")

# ═══════════════════════════════════════════════════════════════════════════════
# PART E: How Trainer Wires It All Together (The Hidden Plumbing)
# ═══════════════════════════════════════════════════════════════════════════════
# When you pass a dataset + collator to Trainer, here's what happens inside:
#
#   1. Trainer creates a DataLoader from your dataset
#   2. DataLoader calls collator as collate_fn
#   3. Collator pads + creates labels → returns a batch
#   4. Trainer feeds batch to model.forward(**batch)
#   5. Model returns loss (using labels from the collator)
#
# You never see the DataLoader — Trainer builds it for you.
# You never write labels manually — the collator creates them.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART E: Trainer Wires It All Together")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tok.eos_token_id

# Notice: we give Trainer raw HF Dataset + collator. No DataLoader!
with tempfile.TemporaryDirectory() as tmp:
    args = TrainingArguments(
        output_dir=tmp,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds,           # ← HF Dataset (not DataLoader!)
        data_collator=collator_lm,     # ← Collator handles padding + labels
        processing_class=tok,
    )

    # Peek at what Trainer builds internally
    internal_loader = trainer.get_train_dataloader()
    print(f"  Trainer's internal DataLoader:")
    print(f"    type:       {type(internal_loader).__name__}")
    print(f"    batch_size: {internal_loader.batch_size}")
    print(f"    collate_fn: {type(internal_loader.collate_fn).__name__}")
    print(f"  ^ Trainer built the DataLoader for you!")

    result = trainer.train()
    print(f"  Training loss: {result.training_loss:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART F: SFTTrainer Goes Even Further
# ═══════════════════════════════════════════════════════════════════════════════
# trl's SFTTrainer accepts raw text and does tokenization + collation for you:
#
#   SFTTrainer(
#       model=model,
#       train_dataset=ds,       # just needs a "text" column (or messages)
#       # No tokenize step needed
#       # No collator needed — it creates DataCollatorForCompletionOnlyLM
#   )
#
# DataCollatorForCompletionOnlyLM is special:
#   - It masks the PROMPT portion with -100
#   - Only the COMPLETION portion contributes to loss
#   - This is what makes SFT work — you don't train on the question,
#     only on the answer.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART F: SFTTrainer — The Full Shortcut")
print("=" * 70)
print("  SFTTrainer hides even more plumbing:")
print("  ┌─────────────────────────────────────────────────┐")
print("  │  You provide:  HF Dataset with 'text' column    │")
print("  │  SFTTrainer:   tokenizes → collates → trains    │")
print("  │  Collator:     DataCollatorForCompletionOnlyLM   │")
print("  │  Result:       Only trains on completions        │")
print("  └─────────────────────────────────────────────────┘")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PART G: The Cheat Sheet
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHEAT SHEET: The Data Pipeline Stack")
print("=" * 70)
print("""
┌──────────────────────────────────────────────────────────────────┐
│                    THE DATA PIPELINE STACK                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: RAW DATA                                               │
│  ├─ CSV, JSONL, text files, Parquet, database exports           │
│  └─ This is YOUR data                                            │
│                                                                  │
│  Layer 2: DATASET (holding & accessing)                          │
│  ├─ torch.utils.data.Dataset  ← subclass, write __getitem__     │
│  │   - Row-oriented, in-memory, you handle everything            │
│  │   - Use when: custom logic, non-text data, legacy code        │
│  │                                                               │
│  └─ datasets.Dataset (HF)     ← from_dict/load_dataset          │
│      - Column-oriented, Arrow-backed, memory-mapped              │
│      - Has .map(), .filter(), .select() — no subclassing         │
│      - Use when: text/NLP, HF models, 99% of fine-tuning        │
│                                                                  │
│  Layer 3: DATACOLLATOR (preparing batches)                       │
│  ├─ Takes list of examples → returns padded + labeled batch      │
│  ├─ DataCollatorWithPadding       → classification               │
│  ├─ DataCollatorForLanguageModeling → causal LM / MLM            │
│  ├─ DataCollatorForSeq2Seq        → translation, summarization   │
│  └─ DataCollatorForCompletionOnlyLM → SFT (masks prompt)        │
│                                                                  │
│  Layer 4: DATALOADER (batching & shuffling)                      │
│  ├─ Wraps Dataset, uses Collator as collate_fn                   │
│  ├─ Handles batch_size, shuffle, num_workers                     │
│  └─ Trainer builds this for you — you rarely touch it            │
│                                                                  │
│  Layer 5: TRAINER (orchestration)                                │
│  ├─ Trainer:     you provide dataset + collator                  │
│  ├─ SFTTrainer:  you provide dataset, it handles everything      │
│  └─ DPO/GRPO:    you provide dataset in expected format          │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  WHAT YOU ACTUALLY NEED TO DO:                                   │
│                                                                  │
│  Causal LM:  HF Dataset + DataCollatorForLanguageModeling        │
│  SFT:        HF Dataset with 'text' column → SFTTrainer          │
│  DPO:        HF Dataset with prompt/chosen/rejected → DPOTrainer │
│  GRPO:       HF Dataset with prompt column → GRPOTrainer         │
│                                                                  │
│  The Trainer handles DataLoader creation. Don't build your own.  │
└──────────────────────────────────────────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════════════
# PART H: Common Gotchas
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("COMMON GOTCHAS")
print("=" * 70)
print("""
1. "I padded in __getitem__ / .map() — why do I need a collator?"
   → You don't, but you're wasting compute. Fixed padding means every
     sequence is max_length even if the batch's longest is 20 tokens.
     Collators do DYNAMIC padding — pad to the batch's longest only.

2. "I wrote a PyTorch Dataset — how do I use it with Trainer?"
   → Just pass it. Trainer accepts anything with __len__ + __getitem__.
     But HF Dataset is almost always simpler. Use .from_dict() instead.

3. "Why does SFTTrainer say 'text' column not found?"
   → SFTTrainer expects a column named 'text' (or chat-format 'messages').
     Rename your column: ds = ds.rename_column("content", "text")

4. "My model trains but loss doesn't decrease"
   → Check that labels are correct. For causal LM, the collator should
     set labels = input_ids with -100 on padding. If you set labels
     yourself AND use a collator, they may conflict.

5. "What's the difference between HF Dataset and PyTorch Dataset?"
   → HF Dataset: column-oriented, Arrow-backed, has .map()/.filter(),
     memory-mapped (handles datasets larger than RAM).
     PyTorch Dataset: row-oriented, in-memory, you subclass it.
     For LLM fine-tuning, always use HF Dataset.

6. "When do I need to write a custom collator?"
   → Almost never for standard fine-tuning. The built-in collators
     cover causal LM, MLM, seq2seq, and SFT. You'd only write one
     for custom loss masking or non-standard batch structures.
""")

print("Dataset ecosystem walkthrough OK!")
