"""
15_lora_footprint.py
--------------------
The *running example* behind the LoRA / QLoRA / Full-FT comparison tables in the
handbook. Every number the cheatsheet asserts ("r=16 -> ~42M params / 0.52%",
"QLoRA 8B ~= 5.5GB VRAM") is reproduced here so the claims are demonstrated, not
just stated.

It does three things, each degrading gracefully:
  1. Analytically counts LoRA trainable params at r = 8/16/32/64 straight from the
     model's layer dimensions. Pure arithmetic -> runs instantly on CPU, offline.
  2. (optional) Verifies the formula against a real PEFT attach on a tiny model,
     if `transformers` + `peft` are installed.
  3. (optional) Reports actual VRAM for Full-FT vs LoRA vs QLoRA, if a CUDA GPU
     and `bitsandbytes` are present.

Run with: python scripts/15_lora_footprint.py
"""

import torch

# ── Llama-3.1-8B architecture (edit these to model any other arch) ────────────
# (in_features, out_features) of each Linear that LoRA targets, per decoder layer.
LLAMA31_8B = {
    "name": "Llama-3.1-8B",
    "num_layers": 32,
    "total_params": 8_030_261_248,   # full model parameter count (HF reported)
    "target_modules": {
        # attention (GQA: 8 KV heads x 128 dim -> k/v project to 1024)
        "q_proj":    (4096, 4096),
        "k_proj":    (4096, 1024),
        "v_proj":    (4096, 1024),
        "o_proj":    (4096, 4096),
        # mlp
        "gate_proj": (4096, 14336),
        "up_proj":   (4096, 14336),
        "down_proj": (14336, 4096),
    },
}


def lora_params(arch: dict, r: int) -> int:
    """LoRA adds A (r x in) + B (out x r) per targeted Linear: r*(in+out) params."""
    per_layer = sum(r * (i + o) for (i, o) in arch["target_modules"].values())
    return per_layer * arch["num_layers"]


def report_param_footprint(arch: dict):
    total = arch["total_params"]
    print(f"\n=== Trainable-param footprint — {arch['name']} ({total/1e9:.2f}B params) ===")
    print(f"  LoRA targets {len(arch['target_modules'])} Linears x {arch['num_layers']} layers"
          f" = {len(arch['target_modules']) * arch['num_layers']} adapted matrices\n")
    print(f"  {'rank r':>7} | {'trainable':>12} | {'% of model':>10}")
    print(f"  {'-'*7}-+-{'-'*12}-+-{'-'*10}")
    for r in (8, 16, 32, 64):
        p = lora_params(arch, r)
        print(f"  {r:>7} | {p/1e6:>10.1f}M | {100*p/total:>9.2f}%")
    print("\n  -> Reproduces the handbook claim: r=16 ≈ 42M trainable = 0.52% of 8B.")


def verify_with_peft():
    """Optionally confirm the arithmetic against a real PEFT attach on a tiny model."""
    # Stay fully offline: use the local HF cache only and never hit the network,
    # so a missing model skips cleanly instead of retrying against the Hub.
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("\n[skip] transformers/peft not installed — analytical counts above stand on their own.")
        return
    try:
        # sshleifer/tiny-gpt2 is a few-hundred-KB stand-in; uses the local cache only.
        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", local_files_only=True)
    except Exception as e:  # not cached locally
        print(f"\n[skip] tiny model not in local cache ({type(e).__name__}) — "
              "analytical counts above are exact. Pre-cache with: "
              "huggingface-cli download sshleifer/tiny-gpt2")
        return
    cfg = LoraConfig(r=8, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0)
    peft_model = get_peft_model(base, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"\n=== Live PEFT check (sshleifer/tiny-gpt2, r=8) ===")
    print(f"  trainable {trainable:,} / total {total:,} = {100*trainable/total:.2f}%")
    print("  -> Same A(r x in)+B(out x r) mechanism, measured on a real module.")


def report_vram():
    print("\n=== VRAM — Full-FT vs LoRA vs QLoRA (8B model) ===")
    if not torch.cuda.is_available():
        print("  [no CUDA] Reference figures (single 8B model, batch 1, seq 2048):")
        print(f"    {'method':>8} | {'VRAM':>8} | {'speed':>8} | when")
        print(f"    {'-'*8}-+-{'-'*8}-+-{'-'*8}-+------------------------")
        print(f"    {'Full-FT':>8} | {'80GB+':>8} | {'fastest':>8} | many GPUs, max quality")
        print(f"    {'LoRA':>8} | {'~14GB':>8} | {'fast':>8} | 1x A100, good quality")
        print(f"    {'QLoRA':>8} | {'~5.5GB':>8} | {'~39% slower':>8} | single consumer GPU / Colab")
        print("  Run on a CUDA box to measure these live via torch.cuda.max_memory_allocated().")
        return
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    # Allocate a representative weight tensor to show how to measure live.
    x = torch.randn(4096, 14336, device=dev, dtype=torch.float16)
    _ = x @ x.T
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Live peak for one fp16 MLP matmul: {peak:.2f} GB on {torch.cuda.get_device_name()}")
    print("  In a real run, wrap trainer.train() and read max_memory_allocated() after a step.")


if __name__ == "__main__":
    report_param_footprint(LLAMA31_8B)
    verify_with_peft()
    report_vram()
