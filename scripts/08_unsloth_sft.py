"""
08_unsloth_sft.py
-----------------
Full Unsloth + LoRA supervised fine-tuning workflow.
Mirrors the mlabonne/sft-llama3 blog post pattern.

Steps:
  1. FastLanguageModel.from_pretrained (4-bit quantized)
  2. get_peft_model → LoRA adapters
  3. Apply chat template to dataset
  4. SFTTrainer → train
  5. Inference
  6. Save adapters / merged / GGUF

NOTE: Requires Unsloth, CUDA GPU, and internet (for model download).
Install: uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Run with: python scripts/08_unsloth_sft.py
"""

import torch

# ── Unsloth import guard ──────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

if not UNSLOTH_AVAILABLE:
    print("Unsloth not installed. Install with:")
    print('  uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    print("\nShowing code structure only (will not run).")

if not torch.cuda.is_available():
    print("GPU not available. Unsloth requires CUDA. Showing code structure only.")

# ── Script continues even without GPU so you can read the full pattern ────────

print("""
=== Unsloth SFT — Full Workflow ===

# ── Step 1: Load 4-bit quantized model ───────────────────────────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # auto-detect (bf16 on Ampere+)
    load_in_4bit=True,   # QLoRA: 4-bit base + fp16 adapters
)

# ── Step 2: Add LoRA adapters ─────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                            # rank (4, 8, 16, 32, 64)
    target_modules=[                 # modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                   # usually == r
    lora_dropout=0.0,                # 0 is optimized in Unsloth
    bias="none",                     # "none" recommended
    use_gradient_checkpointing="unsloth",  # saves VRAM
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# ── Step 3: Apply chat template + format dataset ──────────────────────────────
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def format_prompt(examples):
    conversations = examples["conversations"]   # list of [{"role":..,"content":..}]
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]
    return {"text": texts}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(format_prompt, batched=True)

# ── Step 4: SFTTrainer ────────────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)

# Train on assistant responses only (ignore user/system tokens in loss)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\\n\\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\\n\\n",
)

trainer_stats = trainer.train()

# ── Step 5: Inference ─────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Explain LoRA in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=streamer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

# ── Step 6: Save ──────────────────────────────────────────────────────────────

# Option A: Save LoRA adapters only (small, ~50-200MB)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Option B: Merge adapters into base model (16-bit, full size ~16GB)
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# Option C: Export to GGUF for Ollama / llama.cpp
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
# Then run: ollama create my-model -f Modelfile
""")

if UNSLOTH_AVAILABLE and torch.cuda.is_available():
    print("\nUnsloth + GPU available — uncomment and run the code above!")
else:
    print("\nCode reference printed. Install Unsloth + use a GPU to run.")
