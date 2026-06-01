# Case Study — Fine-Tune a Small LLM for a Niche Domain, Then Run It at the Edge

A single, end-to-end, **runnable and reproducible** study: take a model from a raw Wikipedia corpus
through **continued pretraining (CPT) → supervised fine-tuning (SFT) → 4-bit on-device deployment**,
on a neutral domain (**computational / quantum chemistry**), with two models — **SmolLM2-135M** (edge)
and **SmolLM3-3B** (capability). Every number in the book chapter is produced by these scripts.

> The narrative writeup is the **"Case Study" chapter** in `../docs/handbook.html` (`#casestudy`).
> Recipe + literature: [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) · debugging/tricks: [`PITFALLS.md`](PITFALLS.md).

## Quick start
```bash
# one command — runs §1→§13, streams live logs, prints a results digest:
bash run.sh                 # SmolLM2-135M  (bf16; .venv-rl)
bash run.sh smollm3         # SmolLM3-3B    (QLoRA 4-bit; .venv)
MODE=trial bash run.sh smollm3      # fast smoke test (caps data + steps)
bash run.sh smollm3 06 07           # only specific sections
```
- **Logs**: `logs/<model>/NN_*.log` (live-streamed) · **digest**: `logs/<model>/SUMMARY.md`
- **Prerequisites**: a CUDA GPU; the two virtualenvs (`.venv` = Unsloth+bitsandbytes+trl0.24,
  `.venv-rl` = trl1.x); **Ollama** for Part B deploy/edge/tool; for SmolLM3 GGUF, a one-time
  `git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp` + `pip install -r .../requirements-convert_hf_to_gguf.txt`, then prefix runs with `LLAMA_CPP=~/llama.cpp`.
- `run.sh` auto-sets `HF_HUB_OFFLINE=1` once models are cached (avoids a transformers phone-home crash).

## Layout
```
case_study/
  run.sh                         one-command end-to-end runner (+ live logs + digest)
  README.md  RESEARCH_NOTES.md  PITFALLS.md
  scripts/    config.py  utils.py  01..15_*.py     ← the runnable pipeline (source of truth)
  notebooks/  01..07_*.ipynb                       ← teaching twins for Part A
  data/  outputs/  logs/                           ← generated (gitignored)
```
`run.sh` invokes `scripts/`; you can also run a script directly, e.g. `../.venv/bin/python scripts/03_cpt.py`.

## Scripts (`scripts/`) — the runnable pipeline
| § | Script | What it does | Env | Notebook? |
|---|--------|--------------|-----|-----------|
| 1 | `01_build_corpus.py` | Wikipedia → cleaned CPT corpus; **equation/LaTeX handling**, dedup, manifest | either | ✅ |
| 2 | `02_data_availability.py` | hand-authored Q&A; quantifies the ~99× raw-vs-SFT data asymmetry | either | ✅ |
| 3 | `03_cpt.py` | **CPT** — full causal loss, LoRA incl. `embed_tokens`/`lm_head`; HF + Unsloth backends | both | ✅ |
| 4 | `04_cpt_base_vs_instruct.py` | CPT from base vs instruct + **catastrophic-forgetting** smoke test | per model | ✅ |
| 5 | `05_sft.py` | **SFT** — completion-only loss (prints the unmasked-token fraction) | per model | ✅ |
| 6 | `06_base_vs_instruct_sweep.py` | centerpiece: base-vs-instruct over SFT-set size; perplexity **and** keyword recall | per model | ✅ |
| 7 | `07_eval.py` | scorecard before/after SFT (domain ppl, completion ppl, recall, generations) | per model | ✅ |
| 8 | `08_unsloth_vs_hf.py` | HF vs Unsloth speed/VRAM head-to-head (135M) | both | — |
| 9 | `09_merge_and_gguf.py` | merge LoRA → F16 GGUF (Ollama quantizes to Q4_K_M, <2 GB) | per model | — |
| 10 | `10_ollama_deploy.py` | import into Ollama + query the local API (tok/s) | per model | — |
| 11 | `11_edge_benchmark.py` | on-device footprint + speed (reproduce on a Raspberry Pi with the same code) | per model | — |
| 12 | `12_harness.py` | agent tool-loop + lm-evaluation-harness recipe | per model | — |
| 13 | `13_smollm3_toolcall.py` | tool-calling on our fine-tuned models (native API + capability probe) | per model | — |
| 14 | `14_compare_cpt_vs_sft.py` | **CPT-only vs SFT side by side** (visual + numerical) | per model | — |
| 15 | `15_equation_probe.py` | does the model reproduce domain **equations** in LaTeX? (base vs +CPT) | per model | — |
| — | `config.py` | single source of truth: models, seeds, RUN_MODE, hyperparameters | — | — |
| — | `utils.py` | shared helpers: model loading (4-bit/bf16), SFT model build, generation, metrics | — | — |

Notebooks (`notebooks/`) mirror the Part-A teaching narrative (§1–§7) and import the corresponding
script so code never drifts; Part-B (§8–§15) runs via `run.sh` / direct script calls.

## Data & wrangling
- **Corpus**: 41 Wikipedia pages (comp/quantum chemistry), ~112k tokens, CC BY-SA (attributed in `data/corpus/manifest.json`).
- **Equations**: Wikipedia plaintext double-renders math (glyph dump + `{\displaystyle …}` LaTeX). We extract the LaTeX (balanced-brace scan), strip the glyph soup, NFKC-normalize, and dedup. **1,149 equations** kept as inline `$LaTeX$`. **No new tokens added** (LaTeX is ASCII). §15 confirms the model reproduces equations cleanly.
- **Embeddings**: CPT's LoRA targets **include `embed_tokens`+`lm_head`** (SFT's don't) so the embedding/output layers adapt to the new-domain token distribution, trained at a smaller `embedding_learning_rate` (5e-6 vs 5e-5).
- **SFT data**: 32 hand-authored prompt→completion Q&A (`data/sft/seed_qa.jsonl`) — deliberately small (that scarcity is the point).

## Metrics (what they mean, why chosen)
- **domain perplexity** — exp(mean NLL) on held-out domain text; lower = fits the domain; measures **CPT**.
- **completion perplexity** — perplexity on the answer tokens only (prompt masked); mirrors the SFT objective; measures **SFT**.
- **keyword recall** — fraction of gold-answer content words in the generation; **format-robust** (the fair base-vs-instruct signal, immune to the chat-vs-plain confound); no LLM-as-judge, for reproducibility.
- **footprint / tok/s** — quantized GGUF size (≈ RAM) + Ollama throughput; the **edge** metrics.
- 135M/3B-on-tiny-data are small → report **relative** effects, not absolute SOTA.

## Reproducibility
- `config.py` holds all knobs; seeds set for python/numpy/torch/transformers; env recorded into each metrics JSON.
- `RUN_MODE` (env `CASE_STUDY_MODE` or `set_mode`): **trial** caps data+steps for a fast plumbing check; **full** is the real run. Notebooks expose the same top-level flag.
- Outputs are **model-scoped** (`outputs/<model_key>/`); `data/`, `outputs/`, `logs/` are gitignored (regenerable).
