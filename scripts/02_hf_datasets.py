"""
02_hf_datasets.py
-----------------
Working HuggingFace Datasets examples: load, inspect, map, filter, split, format, push.
Run with: python scripts/02_hf_datasets.py
"""

from datasets import Dataset, DatasetDict, concatenate_datasets

# ── 1. Create from dict (no internet needed) ──────────────────────────────────
print("=== 1. Create from dict ===")
raw = {
    "text": [
        "The quick brown fox jumps over the lazy dog.",
        "PyTorch makes deep learning fun and fast.",
        "HuggingFace datasets are easy to use.",
        "Fine-tuning LLMs is now accessible to everyone.",
        "LoRA reduces memory usage drastically.",
    ],
    "label": [0, 1, 1, 1, 1],
}
ds = Dataset.from_dict(raw)
print(ds)
print("features:", ds.features)
print("first row:", ds[0])

# ── 2. Inspect ────────────────────────────────────────────────────────────────
print("\n=== 2. Inspect ===")
print("num rows:", len(ds))
print("column names:", ds.column_names)
print("first 2 rows:", ds[:2])

# ── 3. map() — tokenise / transform ──────────────────────────────────────────
print("\n=== 3. map() ===")

def add_word_count(example):
    example["word_count"] = len(example["text"].split())
    return example

ds = ds.map(add_word_count)
print("After map:", ds[:2])

# Batched map (faster for tokenisers)
def upper_batch(batch):
    batch["text_upper"] = [t.upper() for t in batch["text"]]
    return batch

ds = ds.map(upper_batch, batched=True, batch_size=3)
print("After batched map:", ds[0]["text_upper"])

# ── 4. filter + select ────────────────────────────────────────────────────────
print("\n=== 4. filter + select ===")
positive = ds.filter(lambda x: x["label"] == 1)
print("positive examples:", len(positive))
subset = ds.select([0, 2, 4])
print("select([0,2,4]):", subset["label"])

# ── 5. rename + remove columns ───────────────────────────────────────────────
print("\n=== 5. rename + remove ===")
ds2 = ds.rename_column("text", "sentence")
print("renamed columns:", ds2.column_names)
ds3 = ds.remove_columns(["text_upper", "word_count"])
print("after remove:", ds3.column_names)

# ── 6. train / test split ─────────────────────────────────────────────────────
print("\n=== 6. train_test_split ===")
splits = ds3.train_test_split(test_size=0.4, seed=42)
print(splits)
print("train:", len(splits["train"]), "  test:", len(splits["test"]))

# ── 7. DatasetDict + concatenate ──────────────────────────────────────────────
print("\n=== 7. DatasetDict + concatenate ===")
dd = DatasetDict({"train": splits["train"], "test": splits["test"]})
print(dd)
combined = concatenate_datasets([dd["train"], dd["test"]])
print("combined:", len(combined))

# ── 8. set_format for PyTorch ─────────────────────────────────────────────────
print("\n=== 8. set_format ===")
import torch
ds4 = ds3.with_format("torch", columns=["label"])
item = ds4[0]
print("label tensor:", item["label"], "  type:", type(item["label"]))

# ── 9. Save + reload ──────────────────────────────────────────────────────────
print("\n=== 9. Save + reload ===")
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "my_dataset")
    ds3.save_to_disk(path)
    from datasets import load_from_disk
    loaded = load_from_disk(path)
    print("reloaded:", loaded)

# ── 10. Load from Hub (online only) ──────────────────────────────────────────
print("\n=== 10. Load from Hub (skipped offline) ===")
print("  Example (run when online):")
print("  from datasets import load_dataset")
print("  ds = load_dataset('tatsu-lab/alpaca', split='train')")
print("  ds = load_dataset('openai/gsm8k', 'main', split='train')")

print("\nAll HF Datasets basics OK!")
