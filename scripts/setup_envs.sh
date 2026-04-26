#!/usr/bin/env bash
# setup_envs.sh — Create both venvs from scratch
# Run from repo root:  bash scripts/setup_envs.sh
set -e

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_VER="torch==2.7.0"
VISION_VER="torchvision==0.22.0"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   LLM Fine-Tuning — Environment Setup               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Two venvs are needed because Unsloth pins trl<0.9,"
echo "which conflicts with DPO/ORPO/GRPO (need trl>=1.0)."
echo ""

# ── .venv — Unsloth + SFT (scripts 01-04, 08) ────────────────────────────────
echo "━━━ Creating .venv (Unsloth + SFT) ━━━"
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  --index-url "$TORCH_INDEX" "$TORCH_VER" "$VISION_VER"
uv pip install --python .venv/bin/python \
  "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  transformers datasets "peft>=0.10" accelerate bitsandbytes evaluate \
  boto3 mergekit llm-blender
echo "✓ .venv ready"

# ── .venv-rl — DPO / ORPO / GRPO (scripts 05-07) ────────────────────────────
echo ""
echo "━━━ Creating .venv-rl (DPO/ORPO/GRPO) ━━━"
uv venv .venv-rl --python 3.12
uv pip install --python .venv-rl/bin/python \
  --index-url "$TORCH_INDEX" "$TORCH_VER" "$VISION_VER"
uv pip install --python .venv-rl/bin/python \
  "trl>=1.0" "transformers>=4.38" datasets "peft>=0.10" accelerate \
  evaluate boto3 mergekit
echo "✓ .venv-rl ready"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Usage                                              ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  PyTorch / HF basics / Unsloth SFT:                 ║"
echo "║    .venv/bin/python scripts/01_pytorch_basics.py    ║"
echo "║    .venv/bin/python scripts/08_unsloth_sft.py       ║"
echo "║    .venv/bin/python scripts/verify_all.py           ║"
echo "║                                                      ║"
echo "║  DPO / ORPO / GRPO alignment:                       ║"
echo "║    .venv-rl/bin/python scripts/05_dpo_example.py    ║"
echo "║    .venv-rl/bin/python scripts/06_orpo_example.py   ║"
echo "║    .venv-rl/bin/python scripts/07_grpo_example.py   ║"
echo "╚══════════════════════════════════════════════════════╝"
