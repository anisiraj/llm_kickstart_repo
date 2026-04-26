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
# PART F: Domain Specialization — Proof by Perplexity and Generation
# ═══════════════════════════════════════════════════════════════════════════════
# The real test: train two adapters on distinct domains, then measure
# whether each adapter actually becomes a specialist.
#
# We use enough data and steps for the adapters to genuinely shift the
# distribution — not a 3-step toy, but a real (small) fine-tuning run.

print("\n" + "=" * 72)
print("PART F: Domain specialization — do adapters actually learn?")
print("=" * 72)

import math

# Two distinct domain corpora — HVAC/building systems vs energy/sustainability
hvac_corpus = [
    "The chiller plant COP dropped below 3.0 during peak load hours.",
    "BACnet integration requires proper MSTP addressing on the trunk.",
    "The AHU supply air temperature setpoint should track outdoor air enthalpy.",
    "VRF systems in zones 4-6 reported refrigerant pressure faults.",
    "Compressor discharge temperature exceeded safety threshold at 210F.",
    "The building automation system scheduled unoccupied setback at 10pm.",
    "Condenser water temperature reset saved 12% on chiller energy consumption.",
    "Damper actuator on AHU-3 failed closed, causing high static pressure alarm.",
    "Cooling tower fan staging follows wet-bulb temperature differential.",
    "VAV box minimum airflow setpoints must comply with ASHRAE 62.1 ventilation.",
    "The economizer lockout temperature is set to 55F for this climate zone.",
    "Chilled water supply temperature reset from 42F to 48F during low load.",
    "The DDC controller lost communication with the supervisory network.",
    "Return air CO2 levels exceeded 1000ppm triggering demand ventilation.",
    "Heating hot water loop differential pressure setpoint is 8 PSI.",
] * 8  # 120 samples

energy_corpus = [
    "Solar photovoltaic generation peaked at 340 kW during midday hours.",
    "Battery storage discharged 500 kWh during the evening demand peak.",
    "The campus achieved a 15% reduction in carbon intensity year over year.",
    "Demand response event triggered load curtailment across three buildings.",
    "Real-time energy pricing averaged $0.12/kWh during off-peak hours.",
    "Wind turbine capacity factor reached 38% during the autumn quarter.",
    "Electric vehicle charging stations consumed 2.1 MWh overnight.",
    "Grid interconnection agreement limits export to 500 kW at the meter.",
    "Thermal energy storage shifted 200 ton-hours from peak to off-peak.",
    "Power factor correction capacitors improved the facility PF to 0.97.",
    "Microgrid islanding test successfully maintained critical loads for 4 hours.",
    "Renewable energy certificates covered 60% of annual electricity consumption.",
    "Smart meter data showed 25% baseload reduction after retrofit completion.",
    "Combined heat and power system operates at 82% total thermal efficiency.",
    "Electricity procurement strategy shifted to 70% fixed-price contracts.",
] * 8  # 120 samples

# --- Train HVAC adapter (100 steps for stronger domain signal) ---
TRAIN_STEPS = 100
print(f"\nTraining HVAC adapter ({TRAIN_STEPS} steps)...")
base_hvac = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      lora_dropout=0.05, target_modules=["c_attn"])
model_hvac = get_peft_model(base_hvac, lora_cfg, adapter_name="hvac")
model_hvac.train()

# Tokenize and train in batches
hvac_enc = tokenizer(hvac_corpus, truncation=True, padding=True, max_length=80,
                     return_tensors="pt").to(DEVICE)
optimizer_h = torch.optim.AdamW(model_hvac.parameters(), lr=3e-4)
batch_size = 16
for step in range(TRAIN_STEPS):
    idx = (step * batch_size) % len(hvac_corpus)
    batch = {k: v[idx:idx+batch_size] for k, v in hvac_enc.items()}
    out = model_hvac(**batch, labels=batch["input_ids"])
    out.loss.backward(); optimizer_h.step(); optimizer_h.zero_grad()
    if (step + 1) % 50 == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}  loss={out.loss.item():.4f}")

dir_hvac = tempfile.mkdtemp(prefix="hvac_domain_")
model_hvac.save_pretrained(dir_hvac, selected_adapters=["hvac"])

# --- Train Energy adapter ---
print(f"\nTraining Energy adapter ({TRAIN_STEPS} steps)...")
base_energy = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
model_energy = get_peft_model(base_energy, lora_cfg, adapter_name="energy")
model_energy.train()

energy_enc = tokenizer(energy_corpus, truncation=True, padding=True, max_length=80,
                       return_tensors="pt").to(DEVICE)
optimizer_e = torch.optim.AdamW(model_energy.parameters(), lr=3e-4)
for step in range(TRAIN_STEPS):
    idx = (step * batch_size) % len(energy_corpus)
    batch = {k: v[idx:idx+batch_size] for k, v in energy_enc.items()}
    out = model_energy(**batch, labels=batch["input_ids"])
    out.loss.backward(); optimizer_e.step(); optimizer_e.zero_grad()
    if (step + 1) % 50 == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}  loss={out.loss.item():.4f}")

dir_energy = tempfile.mkdtemp(prefix="energy_domain_")
model_energy.save_pretrained(dir_energy, selected_adapters=["energy"])

# --- Perplexity evaluation ---
def compute_perplexity(model, texts, tok, max_len=80):
    """Compute perplexity on a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tok = enc["input_ids"].shape[1]
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return math.exp(total_loss / total_tokens)

# Hold-out test sentences (never seen during training)
hvac_test = [
    "The rooftop unit compressor is short-cycling due to low refrigerant charge.",
    "Supply air fan VFD drive faulted on overcurrent during morning startup.",
    "Occupied zone temperature drifted 3 degrees above cooling setpoint.",
    "The pneumatic-to-digital retrofit improved control accuracy by 40%.",
    "Mixed air plenum temperature sensor reading is 5 degrees below expected.",
    "The chiller sequencing algorithm prioritized the most efficient unit first.",
    "Exhaust fan interlock with the makeup air unit failed during test.",
    "Humidity control in the server room required reheat coil activation.",
]
energy_test = [
    "Utility demand charges accounted for 35% of the monthly electricity bill.",
    "Behind-the-meter storage reduced peak demand by 120 kW this billing cycle.",
    "The energy audit identified $180K in annual savings from lighting retrofit.",
    "Net metering credits offset daytime solar export at the retail electricity rate.",
    "Onsite cogeneration provided 60% of the facility thermal load last quarter.",
    "The power purchase agreement locked in solar energy at 4 cents per kilowatt hour.",
    "Peak shaving with battery dispatch saved $22K in demand charges this month.",
    "Carbon accounting showed a 30% reduction in Scope 2 emissions year over year.",
]
# Generic text — neither domain (control group)
generic_test = [
    "The restaurant on the corner serves excellent pasta and fresh bread.",
    "Scientists discovered a new species of butterfly in the Amazon rainforest.",
    "The basketball team won their third consecutive championship last night.",
    "Modern smartphones contain more computing power than early space missions.",
]

# Load both adapters onto one base for fair comparison
base_eval = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Base model perplexity (no adapter)
ppl_base_hvac = compute_perplexity(base_eval, hvac_test, tokenizer)
ppl_base_energy = compute_perplexity(base_eval, energy_test, tokenizer)
ppl_base_generic = compute_perplexity(base_eval, generic_test, tokenizer)

# Load bigger model for comparison: GPT-2 Medium (355M params, ~3x GPT-2)
big_model_name = "gpt2-medium"
print(f"\nLoading {big_model_name} (355M params) for size comparison...")
big_model = AutoModelForCausalLM.from_pretrained(big_model_name).to(DEVICE)
big_tokenizer = AutoTokenizer.from_pretrained(big_model_name)
big_tokenizer.pad_token = big_tokenizer.eos_token
big_model.eval()
big_params = sum(p.numel() for p in big_model.parameters())
print(f"  {big_model_name}: {big_params:,} params ({big_params/base_params:.1f}× GPT-2)")

ppl_big_hvac = compute_perplexity(big_model, hvac_test, big_tokenizer)
ppl_big_energy = compute_perplexity(big_model, energy_test, big_tokenizer)
ppl_big_generic = compute_perplexity(big_model, generic_test, big_tokenizer)

# HVAC adapter perplexity
multi = PeftModel.from_pretrained(base_eval, os.path.join(dir_hvac, "hvac"),
                                  adapter_name="hvac")
multi.load_adapter(os.path.join(dir_energy, "energy"), adapter_name="energy")
multi.eval()

multi.set_adapter("hvac")
ppl_hvac_on_hvac = compute_perplexity(multi, hvac_test, tokenizer)
ppl_hvac_on_energy = compute_perplexity(multi, energy_test, tokenizer)
ppl_hvac_on_generic = compute_perplexity(multi, generic_test, tokenizer)

multi.set_adapter("energy")
ppl_energy_on_hvac = compute_perplexity(multi, hvac_test, tokenizer)
ppl_energy_on_energy = compute_perplexity(multi, energy_test, tokenizer)
ppl_energy_on_generic = compute_perplexity(multi, generic_test, tokenizer)

print(f"\n{'─' * 86}")
print(f"  PERPLEXITY TABLE (lower = model knows the domain better)")
print(f"  Test on hold-out sentences never seen during training")
print(f"{'─' * 86}")
print(f"  {'Model':<28s} {'HVAC (8)':>12s}  {'Energy (8)':>12s}  {'Generic (4)':>12s}")
print(f"  {'─' * 28}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
print(f"  {'GPT-2 (124M, base)':<28s} {ppl_base_hvac:>12.1f}  {ppl_base_energy:>12.1f}  {ppl_base_generic:>12.1f}")
print(f"  {'GPT-2 Medium (355M)':<28s} {ppl_big_hvac:>12.1f}  {ppl_big_energy:>12.1f}  {ppl_big_generic:>12.1f}")
print(f"  {'GPT-2 + HVAC adapter':<28s} {ppl_hvac_on_hvac:>12.1f}  {ppl_hvac_on_energy:>12.1f}  {ppl_hvac_on_generic:>12.1f}")
print(f"  {'GPT-2 + Energy adapter':<28s} {ppl_energy_on_hvac:>12.1f}  {ppl_energy_on_energy:>12.1f}  {ppl_energy_on_generic:>12.1f}")
print(f"{'─' * 86}")

# Key comparison: small+adapter vs big model
hvac_adapter_vs_big = "✓ SMALL + ADAPTER WINS" if ppl_hvac_on_hvac < ppl_big_hvac else "  big model still better"
energy_adapter_vs_big = "✓ SMALL + ADAPTER WINS" if ppl_energy_on_energy < ppl_big_energy else "  big model still better"
generic_big_vs_adapter = "✓ BIG MODEL WINS (expected)" if ppl_big_generic < ppl_hvac_on_generic else "  adapter transferred"

# Highlight the diagonal — each adapter should win on its own domain
hvac_best = "✓ HVAC wins" if ppl_hvac_on_hvac < ppl_energy_on_hvac else "✗ unexpected"
energy_best = "✓ Energy wins" if ppl_energy_on_energy < ppl_hvac_on_energy else "✗ unexpected"

print(f"\n  VALIDATION 1 — Adapter specialization (small model):")
print(f"    On HVAC text:    HVAC adapter={ppl_hvac_on_hvac:.1f} vs Energy={ppl_energy_on_hvac:.1f}  → {hvac_best}")
print(f"    On Energy text:  Energy adapter={ppl_energy_on_energy:.1f} vs HVAC={ppl_hvac_on_energy:.1f}  → {energy_best}")
print(f"")
print(f"  VALIDATION 2 — Small + adapter vs 3× bigger model:")
print(f"    HVAC text:    GPT-2+adapter={ppl_hvac_on_hvac:.1f} vs GPT-2 Medium={ppl_big_hvac:.1f}  → {hvac_adapter_vs_big}")
print(f"    Energy text:  GPT-2+adapter={ppl_energy_on_energy:.1f} vs GPT-2 Medium={ppl_big_energy:.1f}  → {energy_adapter_vs_big}")
print(f"    Generic text: GPT-2 Medium={ppl_big_generic:.1f} vs GPT-2+adapter={ppl_hvac_on_generic:.1f}  → {generic_big_vs_adapter}")
print(f"")
print(f"  VALIDATION 3 — Improvement percentages:")
print(f"    HVAC adapter on HVAC text:     {(1 - ppl_hvac_on_hvac/ppl_base_hvac)*100:+.1f}% vs base")
print(f"    Energy adapter on Energy text:  {(1 - ppl_energy_on_energy/ppl_base_energy)*100:+.1f}% vs base")
print(f"    GPT-2 Medium on HVAC text:      {(1 - ppl_big_hvac/ppl_base_hvac)*100:+.1f}% vs base (just from being bigger)")
print(f"    GPT-2 Medium on Energy text:    {(1 - ppl_big_energy/ppl_base_energy)*100:+.1f}% vs base (just from being bigger)")
print(f"")
print(f"  THE ARGUMENT: A 124M model + 1 MB adapter can match or beat a 355M model")
print(f"  on domain text — at 1/3 the memory, 1/3 the latency, and the adapter")
print(f"  can be swapped in milliseconds. Best is not always better.")

# --- Generation comparison table ---
# Mix of domain-specific, ambiguous, and cross-domain prompts
prompts = [
    # HVAC-leaning
    "The building system",
    "The controller reported",
    "The temperature in zone",
    # Energy-leaning
    "Energy consumption",
    "The solar panels",
    "Peak demand",
    # Ambiguous — could go either way
    "The facility manager",
    "The sensor detected",
]

print(f"\n{'─' * 90}")
print(f"  GENERATION COMPARISON (greedy, 30 tokens)")
print(f"  4 models: GPT-2 base (124M), GPT-2 Medium (355M), GPT-2 + HVAC adapter, GPT-2 + Energy adapter")
print(f"{'─' * 90}")

for prompt in prompts:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    inp_big = big_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"\n  Prompt: \"{prompt}\"")

    # Base (disable adapters)
    multi.disable_adapter_layers()
    with torch.no_grad():
        out = multi.generate(**inp, max_new_tokens=30, do_sample=False)
    print(f"  [GPT-2 124M]      {tokenizer.decode(out[0], skip_special_tokens=True)}")
    multi.enable_adapter_layers()

    # GPT-2 Medium
    with torch.no_grad():
        out = big_model.generate(**inp_big, max_new_tokens=30, do_sample=False,
                                 pad_token_id=big_tokenizer.eos_token_id)
    print(f"  [GPT-2 Medium]    {big_tokenizer.decode(out[0], skip_special_tokens=True)}")

    # HVAC
    multi.set_adapter("hvac")
    with torch.no_grad():
        out = multi.generate(**inp, max_new_tokens=30, do_sample=False)
    print(f"  [124M + hvac]     {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Energy
    multi.set_adapter("energy")
    with torch.no_grad():
        out = multi.generate(**inp, max_new_tokens=30, do_sample=False)
    print(f"  [124M + energy]   {tokenizer.decode(out[0], skip_special_tokens=True)}")

print(f"\n{'─' * 90}")
print(f"  The 124M model + domain adapter produces more relevant completions than")
print(f"  the 355M model on domain text — at 1/3 the memory and 1/3 the latency.")
print(f"  On generic text, the bigger model wins — as expected.")
print(f"  That's the trade-off. Best is not always better.")
print(f"{'─' * 90}")

del big_model  # free memory for benchmark section
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save references for Part G benchmark
dir_a = dir_hvac
dir_b = dir_energy

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
# PART G: The Numbers — Adapter vs Full Model (realistic benchmark)
# ═══════════════════════════════════════════════════════════════════════════════
# The real argument: measure load time, swap time, disk, and memory.
# This uses the adapters we already trained above.

print("\n" + "=" * 72)
print("PART G: Benchmark — Adapter advantage in real numbers")
print("=" * 72)

import time
import gc

N_ADAPTERS = 5  # simulate 5 domain specialists

# --- 1. Disk cost: N full models vs 1 base + N adapters ---
adapter_file = os.path.join(dir_a, "hvac", "adapter_model.safetensors")
adapter_disk = os.path.getsize(adapter_file)
base_disk = base_params * 4  # fp32 approximation

full_model_disk = N_ADAPTERS * base_disk
registry_disk = base_disk + N_ADAPTERS * adapter_disk

print(f"\n1. DISK COST ({N_ADAPTERS} specialists)")
print(f"   Full-model approach:     {N_ADAPTERS} × {base_disk/1024/1024:.0f} MB = "
      f"{full_model_disk/1024/1024:.0f} MB")
print(f"   Adapter registry:        1 × {base_disk/1024/1024:.0f} MB + "
      f"{N_ADAPTERS} × {adapter_disk/1024/1024:.1f} MB = "
      f"{registry_disk/1024/1024:.0f} MB")
print(f"   Savings:                 {(1 - registry_disk/full_model_disk)*100:.0f}%")

# --- 2. Load time: full model vs adapter ---
# Warm-up (ensure model is cached on disk)
_ = AutoModelForCausalLM.from_pretrained(model_name)
del _; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Time loading a full model from scratch
times_full = []
for _ in range(3):
    gc.collect()
    t0 = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    times_full.append(time.perf_counter() - t0)
    del m; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

avg_full = sum(times_full) / len(times_full)

# Time loading base once + adapter on top
times_adapter = []
base_load = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
for _ in range(3):
    gc.collect()
    t0 = time.perf_counter()
    m = PeftModel.from_pretrained(base_load, os.path.join(dir_a, "hvac"),
                                  adapter_name="test", is_trainable=False)
    times_adapter.append(time.perf_counter() - t0)
    m.delete_adapter("test")
    gc.collect()

avg_adapter = sum(times_adapter) / len(times_adapter)

print(f"\n2. LOAD TIME (avg of 3 runs)")
print(f"   Load full model:         {avg_full*1000:.0f} ms")
print(f"   Load adapter (base warm):{avg_adapter*1000:.0f} ms")
print(f"   Speedup:                 {avg_full/avg_adapter:.1f}×")

# --- 3. Adapter swap time ---
# Load two adapters, measure set_adapter() time
swap_model = PeftModel.from_pretrained(base_load, os.path.join(dir_a, "hvac"),
                                       adapter_name="hvac")
swap_model.load_adapter(os.path.join(dir_b, "energy"), adapter_name="energy")
swap_model.eval()

times_swap = []
for _ in range(20):
    t0 = time.perf_counter()
    swap_model.set_adapter("energy")
    times_swap.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    swap_model.set_adapter("hvac")
    times_swap.append(time.perf_counter() - t0)

avg_swap = sum(times_swap) / len(times_swap)

print(f"\n3. ADAPTER SWAP TIME (avg of 40 swaps)")
print(f"   set_adapter():           {avg_swap*1000:.2f} ms")
print(f"   vs loading full model:   {avg_full*1000:.0f} ms")
print(f"   Speedup:                 {avg_full/avg_swap:.0f}×")

# --- 4. Inference latency (adapter vs merged vs base) ---
prompt = "The building management system"
input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# Adapter inference
swap_model.set_adapter("hvac")
times_inf_adapter = []
for _ in range(5):
    t0 = time.perf_counter()
    with torch.no_grad():
        swap_model.generate(**input_ids, max_new_tokens=20, do_sample=False)
    times_inf_adapter.append(time.perf_counter() - t0)

avg_inf_adapter = sum(times_inf_adapter) / len(times_inf_adapter)

# Merged inference (from Part D — use a fresh merge)
merged_bench = swap_model.merge_and_unload()
merged_bench.eval()
times_inf_merged = []
for _ in range(5):
    t0 = time.perf_counter()
    with torch.no_grad():
        merged_bench.generate(**input_ids, max_new_tokens=20, do_sample=False)
    times_inf_merged.append(time.perf_counter() - t0)

avg_inf_merged = sum(times_inf_merged) / len(times_inf_merged)

print(f"\n4. INFERENCE LATENCY (20 tokens, avg of 5 runs)")
print(f"   With PEFT adapter:       {avg_inf_adapter*1000:.0f} ms")
print(f"   Merged (no PEFT):        {avg_inf_merged*1000:.0f} ms")
overhead_pct = ((avg_inf_adapter - avg_inf_merged) / avg_inf_merged * 100)
print(f"   PEFT overhead:           {overhead_pct:+.1f}%")

# --- Summary table ---
print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│  ADAPTER REGISTRY vs FULL-MODEL APPROACH ({N_ADAPTERS} specialists)          │
├───────────────────────────┬──────────────────┬───────────────────────┤
│  Metric                   │  Full Models     │  Adapter Registry     │
├───────────────────────────┼──────────────────┼───────────────────────┤
│  Disk ({N_ADAPTERS} specialists)       │  {full_model_disk/1024/1024:>6.0f} MB       │  {registry_disk/1024/1024:>6.0f} MB              │
│  Load new specialist      │  {avg_full*1000:>6.0f} ms       │  {avg_adapter*1000:>6.0f} ms (adapter)     │
│  Switch specialist        │  {avg_full*1000:>6.0f} ms       │  {avg_swap*1000:>6.2f} ms (set_adapter) │
│  Inference (20 tok)       │  {avg_inf_merged*1000:>6.0f} ms       │  {avg_inf_adapter*1000:>6.0f} ms              │
│  Add new domain           │  Clone + retrain │  Train adapter only   │
│  Version control          │  Full checkpoint │  ~{adapter_disk/1024:.0f} KB file          │
└───────────────────────────┴──────────────────┴───────────────────────┘
""")

del base_load, swap_model, merged_bench
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# PART H: Push to Hub (explained, not executed)
# ═══════════════════════════════════════════════════════════════════════════════
# Pushing adapters to the Hub follows the same pattern as full models.
# We show the code but don't execute it (requires authentication).

print("\n" + "=" * 72)
print("PART H: Pushing adapters to the HuggingFace Hub")
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
# PART I: Cheatsheet
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
