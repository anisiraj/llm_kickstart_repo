"""
14_adapter_save_load.py
-----------------------
Save, reload, merge, and swap LoRA adapters — the deployment pattern
that makes fine-tuning practical.

One base model (~500 MB) + many adapters (~5 MB each) = many specialized
models without duplicating weights. Adapters live on the HuggingFace Hub
just like full models.

Run with: python scripts/14_adapter_save_load.py

References:
  - PEFT docs: https://huggingface.co/docs/peft
  - PEFT quicktour: https://huggingface.co/docs/peft/quicktour
  - Real public adapter: https://huggingface.co/ybelkada/opt-350m-lora
  - LoRA paper: https://arxiv.org/abs/2106.09685
"""

import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Train a Small LoRA Adapter
# ═══════════════════════════════════════════════════════════════════════════════
# We fine-tune GPT-2 with LoRA on a tiny dataset to create an adapter.
# The point isn't the quality — it's seeing what gets saved.

print("=" * 72)
print("PART A: Train a LoRA adapter on GPT-2")
print("=" * 72)

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Count base model parameters
base_params = sum(p.numel() for p in base_model.parameters())
print(f"\nBase model parameters: {base_params:,}")

# Apply LoRA — targeting attention projections (GPT-2 uses c_attn, c_proj)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # rank — lower = smaller adapter
    lora_alpha=16,              # scaling factor
    lora_dropout=0.05,
    target_modules=["c_attn"],  # GPT-2's fused QKV attention
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.2-0.5% of base model — that's the whole point of LoRA

# Quick training on a toy dataset (3 steps, just to get non-random weights)
texts = [
    "The HVAC system detected a fault in zone 3 and switched to backup mode.",
    "Compressor discharge temperature exceeded threshold, initiating cooldown.",
    "Building management system reports optimal energy efficiency at 92%.",
    "Chiller plant sequencing adjusted for overnight low-occupancy profile.",
]
encodings = tokenizer(texts, truncation=True, padding=True, max_length=64,
                      return_tensors="pt").to(DEVICE)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(3):
    outputs = model(**encodings, labels=encodings["input_ids"])
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Step {step + 1}/3  loss={outputs.loss.item():.4f}")

print("\nAdapter trained (toy example — 3 steps on 4 sentences).")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Save Adapter Locally
# ═══════════════════════════════════════════════════════════════════════════════
# save_pretrained() saves ONLY the adapter, not the base model.
# This is the key insight: adapters are tiny.

print("\n" + "=" * 72)
print("PART B: Save adapter locally — what actually gets written")
print("=" * 72)

save_dir = tempfile.mkdtemp(prefix="lora_adapter_")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)  # good practice: keep tokenizer with adapter

print(f"\nSaved to: {save_dir}")
print("\nFiles created:")
for f in sorted(os.listdir(save_dir)):
    size = os.path.getsize(os.path.join(save_dir, f))
    if size > 1024 * 1024:
        print(f"  {f:40s} {size / 1024 / 1024:.1f} MB")
    elif size > 1024:
        print(f"  {f:40s} {size / 1024:.1f} KB")
    else:
        print(f"  {f:40s} {size} bytes")

# Show the adapter config
import json
with open(os.path.join(save_dir, "adapter_config.json")) as f:
    config = json.load(f)
print(f"\nadapter_config.json contents:")
for key in ["base_model_name_or_path", "r", "lora_alpha", "target_modules",
            "task_type", "peft_type"]:
    if key in config:
        print(f"  {key}: {config[key]}")

# Size comparison
adapter_size = os.path.getsize(
    os.path.join(save_dir, "adapter_model.safetensors")
)
# Base model size (approximate from parameter count * 4 bytes for fp32)
base_size_approx = base_params * 4
print(f"\n┌─────────────────────────────────────┐")
print(f"│  Base model:  ~{base_size_approx / 1024 / 1024:.0f} MB (all params)    │")
print(f"│  Adapter:      {adapter_size / 1024 / 1024:.1f} MB              │")
print(f"│  Ratio:        {adapter_size / base_size_approx * 100:.2f}%               │")
print(f"│                                     │")
print(f"│  10 adapters = 10 specialized models │")
print(f"│  for the cost of 1 base + ~50 MB    │")
print(f"└─────────────────────────────────────┘")

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Reload Adapter onto a Fresh Base Model
# ═══════════════════════════════════════════════════════════════════════════════
# In production: load base model once, then load the adapter you need.
# PeftModel.from_pretrained() — NOT get_peft_model() (that's for training).

print("\n" + "=" * 72)
print("PART C: Reload adapter onto a fresh base model")
print("=" * 72)

# Simulate fresh start: load base model from scratch
fresh_base = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Load the adapter we just saved
reloaded = PeftModel.from_pretrained(
    fresh_base,
    save_dir,
    is_trainable=False,  # inference mode — freeze adapter
)
reloaded.eval()

# Verify it works
prompt = "The building management system"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    output = reloaded.generate(**inputs, max_new_tokens=30, do_sample=False)
print(f"\nPrompt: {prompt}")
print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Merge Adapter into Base Model
# ═══════════════════════════════════════════════════════════════════════════════
# merge_and_unload() bakes the LoRA weights into the base weights.
# Result: a plain transformers model — no PEFT dependency needed at inference.
# Trade-off: you lose the ability to swap adapters.

print("\n" + "=" * 72)
print("PART D: Merge adapter into base model")
print("=" * 72)

print("\nBefore merge:")
print(f"  Model type: {type(reloaded).__name__}")
print(f"  Has adapter: True")

merged_model = reloaded.merge_and_unload()

print(f"\nAfter merge:")
print(f"  Model type: {type(merged_model).__name__}")
print(f"  Has adapter: False (LoRA weights baked into base)")

# Same generation, same result — but no PEFT wrapper
with torch.no_grad():
    output_merged = merged_model.generate(**inputs, max_new_tokens=30,
                                          do_sample=False)
print(f"\nSame prompt after merge: "
      f"{tokenizer.decode(output_merged[0], skip_special_tokens=True)}")

# Save the merged model — this is a full model, not an adapter
merged_dir = tempfile.mkdtemp(prefix="merged_model_")
merged_model.save_pretrained(merged_dir)
merged_size = sum(
    os.path.getsize(os.path.join(merged_dir, f))
    for f in os.listdir(merged_dir) if f.endswith((".safetensors", ".bin"))
)
print(f"\nMerged model size: {merged_size / 1024 / 1024:.1f} MB "
      f"(same as base — adapter is baked in)")

# ═══════════════════════════════════════════════════════════════════════════════
# PART E: Load a Real Adapter from the HuggingFace Hub
# ═══════════════════════════════════════════════════════════════════════════════
# Public adapters on the Hub work exactly like full models.
# Example: ybelkada/opt-350m-lora — a LoRA adapter for OPT-350M.
#
# The adapter_config.json on the Hub stores the base model name,
# so PEFT knows which model the adapter was trained on.
#
# Real-world use: teams push adapters to a private Hub org.
# Deployment pulls base model once, then swaps adapters per request.

print("\n" + "=" * 72)
print("PART E: Load a public adapter from the HuggingFace Hub")
print("=" * 72)

hub_adapter = "ybelkada/opt-350m-lora"
hub_base = "facebook/opt-350m"

print(f"\nLoading base model: {hub_base}")
print(f"Loading adapter:    {hub_adapter}")
print(f"(This downloads ~700 MB base + ~6 MB adapter on first run)\n")

try:
    opt_tokenizer = AutoTokenizer.from_pretrained(hub_base)
    opt_base = AutoModelForCausalLM.from_pretrained(hub_base).to(DEVICE)
    opt_with_adapter = PeftModel.from_pretrained(opt_base, hub_adapter)
    opt_with_adapter.eval()

    prompt = "The future of AI is"
    inputs = opt_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = opt_with_adapter.generate(**inputs, max_new_tokens=30,
                                           do_sample=False)
    print(f"Prompt: {prompt}")
    print(f"Output: {opt_tokenizer.decode(output[0], skip_special_tokens=True)}")
    print(f"\nAdapter loaded from Hub — same pattern as loading from disk.")

    # Show adapter info
    print(f"\nAdapter details:")
    for name in opt_with_adapter.peft_config:
        cfg = opt_with_adapter.peft_config[name]
        print(f"  Name: {name}")
        print(f"  Type: {cfg.peft_type}")
        print(f"  Rank: {cfg.r}")
        print(f"  Target modules: {cfg.target_modules}")

    del opt_base, opt_with_adapter  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    print(f"Hub download skipped (offline or network issue): {e}")
    print("This is expected in air-gapped environments.")

# ═══════════════════════════════════════════════════════════════════════════════
# PART F: Multiple Adapters — One Base, Many Specialists
# ═══════════════════════════════════════════════════════════════════════════════
# The real power: load base model once, hot-swap adapters per request.
# In production: HVAC adapter for building queries, energy adapter for
# efficiency queries, maintenance adapter for fault diagnosis.

print("\n" + "=" * 72)
print("PART F: Multiple adapters on one base model")
print("=" * 72)

# Start fresh with GPT-2
base = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Create two different LoRA configs (simulating two fine-tuning runs)
config_a = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      target_modules=["c_attn"])
config_b = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                      target_modules=["c_attn"])

# Train adapter A (HVAC domain)
model_a = get_peft_model(base, config_a, adapter_name="hvac")
model_a.train()
hvac_texts = [
    "Compressor fault detected in chiller unit 7, switching to standby.",
    "Zone temperature setpoint adjusted to 72F for occupied hours.",
]
enc_a = tokenizer(hvac_texts, truncation=True, padding=True, max_length=64,
                  return_tensors="pt").to(DEVICE)
opt_a = torch.optim.AdamW(model_a.parameters(), lr=5e-4)
for _ in range(3):
    loss = model_a(**enc_a, labels=enc_a["input_ids"]).loss
    loss.backward(); opt_a.step(); opt_a.zero_grad()

# Save adapter A
dir_a = tempfile.mkdtemp(prefix="adapter_hvac_")
model_a.save_pretrained(dir_a, selected_adapters=["hvac"])
print(f"\nAdapter 'hvac' saved to {dir_a}")

# Train adapter B (energy domain) — on same base
# Reset base model first
base2 = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
model_b = get_peft_model(base2, config_b, adapter_name="energy")
model_b.train()
energy_texts = [
    "Peak demand charge reduced by 15% through load-shifting strategy.",
    "Solar generation offset 40% of campus electricity consumption today.",
]
enc_b = tokenizer(energy_texts, truncation=True, padding=True, max_length=64,
                  return_tensors="pt").to(DEVICE)
opt_b = torch.optim.AdamW(model_b.parameters(), lr=5e-4)
for _ in range(3):
    loss = model_b(**enc_b, labels=enc_b["input_ids"]).loss
    loss.backward(); opt_b.step(); opt_b.zero_grad()

# Save adapter B
dir_b = tempfile.mkdtemp(prefix="adapter_energy_")
model_b.save_pretrained(dir_b, selected_adapters=["energy"])
print(f"Adapter 'energy' saved to {dir_b}")

# Now: one base model, load both adapters, swap between them
print(f"\nLoading both adapters onto one base model...")
base_fresh = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Load first adapter (named adapters save into a subdirectory)
multi = PeftModel.from_pretrained(base_fresh, os.path.join(dir_a, "hvac"),
                                  adapter_name="hvac")
# Load second adapter
multi.load_adapter(os.path.join(dir_b, "energy"), adapter_name="energy")

multi.eval()

# List loaded adapters
print(f"Loaded adapters: {list(multi.peft_config.keys())}")

# Swap and generate
prompt = "The system reported"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

multi.set_adapter("hvac")
with torch.no_grad():
    out_hvac = multi.generate(**inputs, max_new_tokens=25, do_sample=False)
print(f"\n[hvac]   {tokenizer.decode(out_hvac[0], skip_special_tokens=True)}")

multi.set_adapter("energy")
with torch.no_grad():
    out_energy = multi.generate(**inputs, max_new_tokens=25, do_sample=False)
print(f"[energy] {tokenizer.decode(out_energy[0], skip_special_tokens=True)}")

print(f"\nSame base model, different adapter = different specialist.")
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  THE ADAPTER REGISTRY PATTERN                                       │
│                                                                     │
│  One base model + named adapters = organizational knowledge registry│
│                                                                     │
│  ┌──────────┐   ┌───────────────┐                                   │
│  │ Base Model│──▶│ hvac-faults   │ ← domain expertise, versioned    │
│  │ (shared)  │──▶│ energy-opt    │ ← independently trainable        │
│  │           │──▶│ maintenance   │ ← auditable, swappable           │
│  │           │──▶│ compliance    │ ← add/retire without touching    │
│  └──────────┘   └───────────────┘   the base model                  │
│                                                                     │
│  Add knowledge  → train a new adapter                               │
│  Retire knowledge → delete the adapter                              │
│  Update knowledge → retrain, push new version to Hub                │
│                                                                     │
│  An org that maintains an adapter registry OWNS its AI knowledge —  │
│  versioned like code, auditable like data, vendor-independent.      │
└─────────────────────────────────────────────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════════════
# PART G: Push to Hub (explained, not executed)
# ═══════════════════════════════════════════════════════════════════════════════
# Pushing adapters to the Hub follows the same pattern as full models.
# We show the code but don't execute it (requires authentication).

print("\n" + "=" * 72)
print("PART G: Pushing adapters to the HuggingFace Hub")
print("=" * 72)

print("""
# ── Push adapter to Hub ──────────────────────────────────────────────
# Requires: huggingface-cli login (or HF_TOKEN env var)

# After training:
model.push_to_hub("your-username/gpt2-hvac-lora")
tokenizer.push_to_hub("your-username/gpt2-hvac-lora")

# What gets uploaded:
#   adapter_config.json        (~1 KB)  — LoRA hyperparameters
#   adapter_model.safetensors  (~5 MB)  — just the LoRA weights
#   tokenizer files            (~2 MB)  — for reproducibility
#
# Total upload: ~7 MB instead of ~500 MB

# ── Pull adapter from Hub ────────────────────────────────────────────
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model = PeftModel.from_pretrained(base, "your-username/gpt2-hvac-lora")

# ── Or use AutoPeftModel (auto-detects base model) ───────────────────
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("your-username/gpt2-hvac-lora")
# ^ reads adapter_config.json → finds base_model_name_or_path → loads both

# ── Private adapters for teams ────────────────────────────────────────
model.push_to_hub("my-org/hvac-fault-adapter", private=True)
# Team members: PeftModel.from_pretrained(base, "my-org/hvac-fault-adapter")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# PART H: Cheatsheet
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("CHEATSHEET: Adapter Lifecycle")
print("=" * 72)
print("""
  ┌─────────────┐     save_pretrained()     ┌──────────────────┐
  │   Train      │ ────────────────────────▶ │  Local Disk      │
  │   (LoRA)     │                           │  adapter_config   │
  │              │     push_to_hub()         │  adapter_model    │
  │              │ ────────────────────────▶ │  (~5 MB)          │
  └─────────────┘                           └──────────────────┘
                                                     │
                      PeftModel.from_pretrained()    │
                   ◀─────────────────────────────────┘
                      (local path OR Hub repo ID)
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         ┌────────────┐  ┌───────────┐  ┌─────────────┐
         │ Inference   │  │  Merge    │  │  Swap       │
         │ (with PEFT) │  │  & Unload │  │  Adapters   │
         │             │  │           │  │             │
         │ model.eval()│  │ No PEFT   │  │ set_adapter │
         │ generate()  │  │ at deploy │  │ ("hvac")    │
         └────────────┘  └───────────┘  └─────────────┘

  ╔═══════════════════════════════════════════════════════════════════╗
  ║  GOTCHAS                                                         ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  • get_peft_model() = create new adapter for TRAINING            ║
  ║    PeftModel.from_pretrained() = LOAD trained adapter            ║
  ║    Mixing them up = random weights or crashes.                   ║
  ║                                                                  ║
  ║  • merge_and_unload() is NOT in-place:                           ║
  ║    model = model.merge_and_unload()  ✓                           ║
  ║    model.merge_and_unload()          ✗ (returns new, ignores)    ║
  ║                                                                  ║
  ║  • Base model version must match. adapter_config.json records    ║
  ║    the base model — load a different one and shapes mismatch.    ║
  ║                                                                  ║
  ║  • load_adapter() does NOT activate it. Call set_adapter() too.  ║
  ║                                                                  ║
  ║  • QLoRA: if you trained with 4-bit base, load with 4-bit base. ║
  ║    Adapter weights are fp32 regardless.                          ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")

print("Done. Run this script in .venv (requires peft, transformers, torch).")
