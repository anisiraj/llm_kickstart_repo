"""
11_custom_data.py
-----------------
"I have my own data" — how to load CSV, JSONL, and plain text files and
convert them into the exact formats expected by each trainer:

  • Causal LM  → {"text": ...}
  • SFT        → {"prompt": ..., "completion": ...} or chat messages
  • DPO / ORPO → {"prompt": ..., "chosen": ..., "rejected": ...}

Uses only built-in Python + datasets + transformers. No GPU, no downloads.

Run with: python scripts/11_custom_data.py
Requirements: datasets, transformers
"""

import os
import csv
import json
import tempfile
from datasets import Dataset, load_dataset

# We'll create temp files to simulate real user data
tmpdir = tempfile.mkdtemp()

# ══════════════════════════════════════════════════════════════════════════════
# PART A: Loading your data
# ���═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART A: Loading data from different file formats")
print("=" * 60)

# ── A1. CSV ───────────────────────────────────────────────────────────────────
print("\n--- A1. CSV file ---")

csv_path = os.path.join(tmpdir, "my_data.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer"])
    writer.writerow(["What is LoRA?", "LoRA adds small trainable matrices to frozen weights."])
    writer.writerow(["What is SFT?", "Supervised fine-tuning on prompt-response pairs."])
    writer.writerow(["What is DPO?", "Direct preference optimization from chosen/rejected pairs."])
    writer.writerow(["What is GRPO?", "Group relative policy optimization with verifiable rewards."])

ds_csv = load_dataset("csv", data_files=csv_path, split="train")
print(f"  loaded {len(ds_csv)} rows from CSV")
print(f"  columns: {ds_csv.column_names}")
print(f"  sample:  {ds_csv[0]}")

# ── A2. JSONL ─────────────────────────────────────────────────────────────────
print("\n--- A2. JSONL file ---")

jsonl_path = os.path.join(tmpdir, "my_data.jsonl")
records = [
    {"instruction": "Explain quantization", "output": "Quantization compresses weights to 4-bit or 8-bit."},
    {"instruction": "What is a chat template?", "output": "A format that structures conversation turns for the model."},
    {"instruction": "Why use LoRA?", "output": "It reduces memory by training only low-rank adapter matrices."},
    {"instruction": "What is overfitting?", "output": "When the model memorizes training data instead of generalizing."},
]
with open(jsonl_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

ds_jsonl = load_dataset("json", data_files=jsonl_path, split="train")
print(f"  loaded {len(ds_jsonl)} rows from JSONL")
print(f"  columns: {ds_jsonl.column_names}")
print(f"  sample:  {ds_jsonl[0]}")

# ── A3. Plain text ────────────────────────────────────────────────────────────
print("\n--- A3. Plain text file ---")

txt_path = os.path.join(tmpdir, "corpus.txt")
with open(txt_path, "w") as f:
    f.write("LoRA injects low-rank matrices into transformer layers.\n")
    f.write("QLoRA adds 4-bit quantization on top of LoRA.\n")
    f.write("SFT is the first step in building an instruction-following model.\n")
    f.write("The learning rate is the most impactful fine-tuning hyperparameter.\n")

ds_txt = load_dataset("text", data_files=txt_path, split="train")
print(f"  loaded {len(ds_txt)} rows from text")
print(f"  columns: {ds_txt.column_names}")
print(f"  sample:  {ds_txt[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B: Converting to trainer-expected formats
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART B: Converting to trainer-expected formats")
print("=" * 60)

# ── B1. Causal LM format ─────────────────────────────────────────────────────
# Trainer + DataCollatorForLanguageModeling expects: {"text": "..."}
# Just raw text — the collator handles labels automatically.

print("\n--- B1. Causal LM format (raw text) ---")
print('  Expected: {"text": "..."}')

# From CSV (Q&A pairs → concatenated text)
ds_causal = ds_csv.map(
    lambda x: {"text": f"Q: {x['question']} A: {x['answer']}"},
    remove_columns=ds_csv.column_names,
)
print(f"  from CSV:  {ds_causal[0]}")

# From plain text — already in the right format!
print(f"  from text: {ds_txt[0]}")

# ── B2. SFT format ───────────────────────────────────────────────────────────
# SFTTrainer accepts several formats. The simplest:
#   Option 1: {"text": "full formatted string"}
#   Option 2: {"prompt": "...", "completion": "..."}  (auto-formatted by trainer)
#   Option 3: chat messages format (for chat models)

print("\n--- B2. SFT format ---")
print('  Option 1: {"text": "<formatted prompt+response>"}')
print('  Option 2: {"prompt": "...", "completion": "..."}')

# From CSV → prompt/completion
ds_sft = ds_csv.map(
    lambda x: {"prompt": x["question"], "completion": x["answer"]},
    remove_columns=ds_csv.column_names,
)
print(f"  from CSV (option 2): {ds_sft[0]}")

# From JSONL → rename columns
ds_sft2 = ds_jsonl.rename_columns({"instruction": "prompt", "output": "completion"})
print(f"  from JSONL (renamed): {ds_sft2[0]}")

# Chat messages format (for chat-tuned models like Llama-3)
print("\n  Option 3: chat messages format")
ds_chat = ds_csv.map(
    lambda x: {
        "messages": [
            {"role": "user", "content": x["question"]},
            {"role": "assistant", "content": x["answer"]},
        ]
    },
    remove_columns=ds_csv.column_names,
)
print(f"  from CSV (chat): {ds_chat[0]}")

# ── B3. DPO / ORPO format ────────────────────────────────────────────────────
# DPOTrainer and ORPOTrainer expect:
#   {"prompt": "...", "chosen": "...", "rejected": "..."}

print("\n--- B3. DPO / ORPO format ---")
print('  Expected: {"prompt": "...", "chosen": "...", "rejected": "..."}')

# Simulate: you have a CSV with good and bad answers
pref_path = os.path.join(tmpdir, "preferences.csv")
with open(pref_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "good_answer", "bad_answer"])
    writer.writerow([
        "What is LoRA?",
        "LoRA adds small trainable low-rank matrices to frozen model weights, reducing memory.",
        "LoRA makes models faster.",
    ])
    writer.writerow([
        "What is SFT?",
        "SFT teaches a model to follow instructions using prompt-response pairs.",
        "SFT is a type of model.",
    ])
    writer.writerow([
        "What is quantization?",
        "Quantization compresses weights to lower bit-widths like 4-bit or 8-bit.",
        "Quantization removes parts of the model.",
    ])

ds_pref_raw = load_dataset("csv", data_files=pref_path, split="train")
ds_dpo = ds_pref_raw.map(
    lambda x: {
        "prompt": x["question"],
        "chosen": x["good_answer"],
        "rejected": x["bad_answer"],
    },
    remove_columns=ds_pref_raw.column_names,
)
print(f"  from CSV: {ds_dpo[0]}")

# ── B4. GRPO format ──────────────────────────────────────────────────────────
# GRPOTrainer expects: {"prompt": "..."}
# Rewards come from reward functions, not from the dataset.
# But you can pass extra columns as kwargs to reward functions.

print("\n--- B4. GRPO format ---")
print('  Expected: {"prompt": "..."}  (reward comes from functions)')

ds_grpo = ds_csv.map(
    lambda x: {"prompt": x["question"], "answer": x["answer"]},
    remove_columns=["question"],
)
print(f"  from CSV: {ds_grpo[0]}")
print("  The 'answer' column is passed to reward_funcs via **kwargs")


# ══════════════════════════════════════════════════════════════════════════════
# PART C: Practical patterns
# ════��═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART C: Practical patterns")
print("=" * 60)

# ── C1. Train/test split ─────────────────────────────────────────────────────
print("\n--- C1. Always split your data ---")
splits = ds_sft.train_test_split(test_size=0.2, seed=42)
print(f"  train: {len(splits['train'])} rows")
print(f"  test:  {len(splits['test'])} rows")
print("  Use test split for eval_dataset to detect overfitting.")

# ── C2. Filtering bad rows ───────────────────────────────────────────────────
print("\n--- C2. Filter empty or short rows ---")
noisy_data = Dataset.from_dict({
    "text": ["Good sentence here.", "", "OK", "Another valid training example."]
})
clean = noisy_data.filter(lambda x: len(x["text"]) > 10)
print(f"  before: {len(noisy_data)} rows → after: {len(clean)} rows")

# ── C3. Deduplication ────────────────────────────────────────────────────────
print("\n--- C3. Remove duplicates ---")
duped = Dataset.from_dict({"text": ["hello", "world", "hello", "foo", "world"]})
seen = set()
def dedup(example):
    if example["text"] in seen:
        return False
    seen.add(example["text"])
    return True
unique = duped.filter(dedup)
print(f"  before: {len(duped)} → after: {len(unique)}")

# ── C4. Saving processed dataset ─────────────────────────────────────────────
print("\n--- C4. Save processed dataset ---")
save_path = os.path.join(tmpdir, "processed_dataset")
ds_sft.save_to_disk(save_path)
print(f"  saved to: {save_path}")

# Reload
from datasets import load_from_disk
reloaded = load_from_disk(save_path)
print(f"  reloaded: {len(reloaded)} rows, columns: {reloaded.column_names}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SUMMARY: Format cheatsheet")
print("=" * 60)
print("""
  Trainer              | Expected columns
  ---------------------+-----------------------------------------------
  Causal LM (Trainer)  | {"text": "raw text"}
  SFT (SFTTrainer)     | {"prompt": "...", "completion": "..."}
                       |   OR {"messages": [{role, content}, ...]}
  DPO (DPOTrainer)     | {"prompt": "...", "chosen": "...", "rejected": "..."}
  ORPO (ORPOTrainer)   | {"prompt": "...", "chosen": "...", "rejected": "..."}
  GRPO (GRPOTrainer)   | {"prompt": "..."}  + reward functions

  Loading functions:
    CSV   → load_dataset("csv",  data_files="path.csv")
    JSONL → load_dataset("json", data_files="path.jsonl")
    Text  → load_dataset("text", data_files="path.txt")
    Dir   → load_dataset("csv",  data_files="data/*.csv")

  Key steps:
    1. Load with load_dataset()
    2. Rename/map columns to match trainer expectations
    3. Filter out empty/short/duplicate rows
    4. Split into train/test
    5. Pass to trainer as train_dataset / eval_dataset
""")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print("Custom data demo OK!")