# CLAUDE.md — llm_kickstart_repo

A hands-on fine-tuning ebook/handbook + runnable scripts. The "book" = `docs/handbook.html` (+ `docs/index.html`, `docs/cheatsheet_*.html`) backed by `scripts/NN_*.py` running examples and `notebooks/`.

## Core rules (non-negotiable)
1. **Every concept must have a running example that is runnable AND reproducible** (seeds set+recorded, versions pinned, artifacts cached, runs top-to-bottom, degrades gracefully offline/CPU).
2. **Benchmarks are measured, not asserted** — replace marketing numbers with what you measured on this hardware (name the GPU). Past finds: AMP 1.52× (not "2×"), batched map 8.8× (not bare "10-100×").
3. **No copyrighted content.** The sibling `../llm_finetuning_book_tutorial/FineTuningLLMs/` book (dvgodoy) and its Yoda/Phi-3 example are off-limits. Use neutral examples; grep `yoda|phi-3` must be 0 before publishing. Attribute Wikipedia (CC BY-SA).

## Environment
- Repo `.venv`: torch 2.7.0+cu128, transformers 5.6.2, datasets 4.8.4, peft 0.19.1, trl 1.2.0, accelerate 1.6.0. Missing: bitsandbytes, unsloth, wikipedia, matplotlib.
- Unsloth pins `trl<0.9` / `xformers` → conflicts with trl 1.x; use a SEPARATE env (`.venv-rl` pattern) for the Unsloth path.
- CUDA: RTX 3080 Ti available.

## ACTIVE WORK — CPT→SFT computational/quantum-chemistry case study
Building a neutral, end-to-end **case-study chapter**: scrape Wikipedia (comp/quantum chem) → CPT → SFT → eval, with **SmolLM2-135M** (base vs `-Instruct`). Experimental workspace lives in **`case_study/`** (run everything there), then distill references/content into `docs/handbook.html` + `notebooks/`.
- 5 research questions, eval design, and full plan: see memory `project-cpt-sft-comp-chem-case-study`.
- Source material to neutralize: `../llm_finetuning_book_tutorial/notes/{sft_unsloth_guide.md, cheatsheet.html}` (the user's own notes).
- Centerpiece experiment: base-init vs instruct-init SFT under small data (2×3 grid {base,instruct}×N∈{10,30,100}).
- **Part B (advanced) — Deploy at the Edge:** merge→GGUF (llama.cpp, Q4_K_M)→Ollama→embedded device (RPi-class, measure tok/s+RAM)→harness (lm-evaluation-harness + minimal agent loop). Research + sources: `case_study/RESEARCH_NOTES.md`.

## Resume pointers
- File-based memory: `~/.claude/projects/-home-deepti-tutorials-llm-kickstart-repo/memory/` (see MEMORY.md).
- Branch `dev` is fully committed + pushed (latest `1b766f8`). Case study rewritten (first-person, measured); FULL SmolLM3 + overnight MiniCPM5/rank-sweep numbers folded in.
- Scripts now §1–§20 in `case_study/scripts/` (new: 17 push-to-HF, 18 token-surgery, 19 synth-data, 20 merge — all tested).
- Site additions: 5 cheatsheets (`docs/cheatsheet_{sft_trainer,chat_templates_data,metrics,deployment,datagen_merge_hf}.html`) + 7 handbook "Deep Dives" chapters + Evaluation format-trap upgrade. See memory `project-cheatsheets-deepdives-hf`.
- HF artifacts published under `anisiraj/` (dataset + 3 adapters live; MiniCPM5 merged was finishing). HF_TOKEN env var is INVALID → use `env -u HF_TOKEN` (cached token); hf.co uploads flaky.
- Superseded: the old cheatsheet-absorption work is now committed (ignore `project-cheatsheet-absorption-status` as "uncommitted").
