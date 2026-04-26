# Fine-tune Llama 3.1 with SFT (Unsloth + QLoRA)

> Summary of: https://huggingface.co/blog/mlabonne/sft-llama3  
> Author: Maxime Labonne

---

## What is Supervised Fine-Tuning (SFT)?

SFT retrains a pre-trained LLM on a smaller dataset of (instruction, answer) pairs to:

- Convert a base model into an instruction-following assistant
- Improve performance on specific tasks or domains
- Adapt tone/style to a use case

**When to use SFT:**
1. Try prompt engineering / RAG first
2. If that doesn't meet quality/cost/latency needs **and** you have instruction data → use SFT
3. Note: SFT works best with knowledge already in the base model — it struggles to inject truly new information

---

## Fine-Tuning Techniques

| Method | VRAM | Quality | Notes |
|--------|------|---------|-------|
| Full Fine-Tuning | Very High | Best | Risk of catastrophic forgetting |
| LoRA | Medium | Near-full | Trains <1% of params; adapters are swappable |
| QLoRA | Low | Slightly lower | 4-bit quantized base + LoRA; ~33% less VRAM, ~39% slower |

**This tutorial uses QLoRA** (ideal for limited GPU memory, e.g., Colab).

---

## Setup

### Install Dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

### Imports

```python
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
```

---

## Step 1 — Load the Base Model (4-bit)

```python
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,   # QLoRA: quantize base model to 4-bit
    dtype=None,          # auto-detect bf16/fp16
)
```

**Why 4-bit?** The model goes from ~16 GB to ~5.4 GB — fits on a free/cheap GPU.

---

## Step 2 — Attach LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                # Rank: size of adapter matrices
    lora_alpha=16,       # Scaling factor (keep = r for simplicity)
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "up_proj", "down_proj", "o_proj", "gate_proj"
    ],
    use_rslora=True,                        # Rank-Stabilized LoRA (better scaling)
    use_gradient_checkpointing="unsloth",   # Saves VRAM
)
```

Result: **42M trainable params out of 8B (0.52%)** — very efficient.

---

## Step 3 — Load & Format the Dataset

**Dataset used:** `mlabonne/FineTome-100k` (100k high-quality instruction samples, ShareGPT format)

```python
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",   # Options: chatml | llama3 | mistral
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        for message in messages
    ]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)
```

### ChatML Format (what it looks like after templating)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

---

## Step 4 — Configure & Run Training

### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | `3e-4` | Standard for LLM fine-tuning |
| `lr_scheduler_type` | `linear` | Gradually decays LR |
| `per_device_train_batch_size` | `8` | Samples per GPU step |
| `gradient_accumulation_steps` | `2` | Effective batch = 16 |
| `num_train_epochs` | `1` | Enough for 100k samples |
| `optim` | `adamw_8bit` | Memory-efficient optimizer |
| `weight_decay` | `0.01` | Regularization |
| `warmup_steps` | `10` | LR ramp-up at start |
| `packing` | `True` | Combine short samples → faster |

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()
```

### Expected Training Time

| GPU | Time |
|-----|------|
| A100 40GB | ~4h 45m (recommended) |
| L4 | ~19h 40m |
| Free T4 | ~47h |

> Tip: Use RunPod / Lambda Labs / Paperspace for cheaper A100 access. Or train on a subset of the data.

---

## Step 5 — Test Inference

```python
model = FastLanguageModel.for_inference(model)  # Enable faster inference mode

messages = [{"from": "human", "value": "Is 9.11 larger than 9.9?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
```

---

## Step 6 — Save / Export the Model

### Option A: Save LoRA adapters only (smallest)
```python
model.save_pretrained("model")
```

### Option B: Merge + save as 16-bit (best quality)
```python
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
# Push to Hub:
model.push_to_hub_merged("your-username/YourModel", tokenizer, save_method="merged_16bit")
```

### Option C: Export as GGUF (for local inference tools)
```python
# Multiple quantization levels in one go
quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
for quant in quant_methods:
    model.push_to_hub_gguf("your-username/YourModel-GGUF", tokenizer, quant)
```

GGUF works with: **llama.cpp, LM Studio, Ollama, text-generation-webui**

---

## Tools Used

| Tool | Role |
|------|------|
| [Unsloth](https://github.com/unslothai/unsloth) | 2x faster training, 60% less VRAM (custom CUDA kernels) |
| TRL `SFTTrainer` | Supervised fine-tuning loop |
| PEFT | LoRA adapter management |
| Bitsandbytes | 4-bit quantization |
| Transformers | Model/tokenizer infrastructure |

> **Unsloth limitation**: Single-GPU only. For multi-GPU, use TRL or Axolotl directly.

---

## What's Next After SFT

1. **Evaluate** — Use Open LLM Leaderboard or LLM AutoEval
2. **Align** — Apply DPO (Direct Preference Optimization) for preference alignment
3. **Quantize** — EXL2, AWQ, GPTQ, HQQ for deployment
4. **Deploy** — Hugging Face Spaces (needs ~20k samples for good chat performance)

---

## Resources

| Resource | Link |
|----------|------|
| Original blog post | https://huggingface.co/blog/mlabonne/sft-llama3 |
| Google Colab notebook | Linked in the blog post |
| LLM Course (full curriculum) | https://github.com/mlabonne/llm-course |
| Dataset used | `mlabonne/FineTome-100k` on HuggingFace |
| Trained model | `mlabonne/FineLlama-3.1-8B` on HuggingFace |
| GGUF quants | `mlabonne/FineLlama-3.1-8B-GGUF` on HuggingFace |