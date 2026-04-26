# LLM Fine-Tuning Kickstart Repo

**A self-sufficient learning kit for LLM fine-tuning** — from raw tensors to a deployed model in the cloud.  
Handbook, cheatsheets, and runnable scripts. All code verified. No fluff.

🌐 **[View the site → anisiraj.github.io/llm_kickstart_repo](https://anisiraj.github.io/llm_kickstart_repo/)**

---

## What's inside

| | |
|---|---|
| 📘 **End-to-End Handbook** | 9 chapters: PyTorch → HF Datasets → Transformers → LoRA/QLoRA → Unsloth → Causal LM → DPO/ORPO/GRPO → SageMaker → Bedrock |
| 🗂 **6 Cheatsheets** | One-page reference cards for PyTorch, HF Datasets, Transformers, Unsloth/LoRA, AWS SageMaker/Bedrock, VSCode shortcuts |
| 🐍 **9 Verified Scripts** | Runnable end-to-end examples, each self-contained |
| 📓 **Jupyter Notebook** | Full walkthrough — SFT, DPO, ORPO, GRPO, inference, cloud deployment |

---

## Learning Philosophy

> **AI-Powered, Self-Driven Nano Bootcamps**

Traditional learning paths front-load syntax and defer building. This repo is built around a different approach:

1. **Grasp the fundamentals** — tensors, loss, tokenizers. These concepts are load-bearing; syntax isn't.
2. **Pick a problem you own** — a domain, a dataset, a frustration. Ownership creates motivation.
3. **Build headfirst with AI** — use an AI assistant to handle boilerplate. You direct; it executes.
4. **Let the problem teach you** — depth arrives on demand, driven by the specific wall you hit.

*AI handles syntax. You own the fundamentals and judgment.*

---

## Scripts

| Script | What it covers | Venv |
|--------|---------------|------|
| `01_pytorch_basics.py` | Tensors, autograd, training loop | `.venv` |
| `02_hf_datasets.py` | Load, map, filter, format | `.venv` |
| `03_hf_transformers.py` | Tokenizer, pipeline, Trainer | `.venv` |
| `04_causal_lm_finetune.py` | GPT-2 causal LM + generation | `.venv` |
| `05_dpo_example.py` | DPO with synthetic preference data | `.venv-rl` |
| `06_orpo_example.py` | ORPO — no reference model needed | `.venv-rl` |
| `07_grpo_example.py` | GRPO with verifiable math rewards | `.venv-rl` |
| `08_unsloth_sft.py` | Full Unsloth SFT workflow | `.venv` |
| `09_sagemaker_train.py` | SageMaker + Bedrock patterns | `.venv` |

---

## Setup

Two virtual environments are required — Unsloth pins `trl<0.9` which conflicts with the DPO/ORPO/GRPO trainers in `trl≥1.0`.

```bash
# One-shot setup (CUDA GPU required)
bash scripts/setup_envs.sh

# Run SFT / foundation scripts
.venv/bin/python scripts/01_pytorch_basics.py

# Run alignment scripts (DPO / ORPO / GRPO)
.venv-rl/bin/python scripts/05_dpo_example.py

# Verify all packages
.venv/bin/python scripts/verify_all.py
```

**Requirements:** CUDA GPU · Python 3.10+ · ~5 GB disk for model downloads

---

## Site

The handbook and cheatsheets are hosted on GitHub Pages.  
Enable it at: **repo → Settings → Pages → Branch: `master`, Folder: `/docs`**

🌐 [anisiraj.github.io/llm_kickstart_repo](https://anisiraj.github.io/llm_kickstart_repo/)

---

## References

Key papers this repo is built on:

- **LoRA** — Hu et al. 2021 · [arxiv/2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA** — Dettmers et al. 2023 · [arxiv/2305.14314](https://arxiv.org/abs/2305.14314)
- **DPO** — Rafailov et al. 2023 · [arxiv/2305.18290](https://arxiv.org/abs/2305.18290)
- **ORPO** — Hong et al. 2024 · [arxiv/2403.07691](https://arxiv.org/abs/2403.07691)
- **GRPO** — Shao et al. 2024 · [arxiv/2402.03300](https://arxiv.org/abs/2402.03300)
- [PyTorch](https://pytorch.org/docs) · [HF Transformers](https://huggingface.co/docs/transformers) · [TRL](https://huggingface.co/docs/trl) · [Unsloth](https://github.com/unslothai/unsloth)
