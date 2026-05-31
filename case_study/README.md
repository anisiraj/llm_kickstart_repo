# Case Study — Fine-Tune a Small LLM for a Niche Domain, Then Deploy It at the Edge

An end-to-end, **runnable and reproducible** walkthrough that takes a 135M model from a raw
Wikipedia corpus all the way to a quantized model running on an embedded device — using
**computational & quantum chemistry** as the neutral domain.

> Every number reported in the eventual book chapter is **measured on the hardware named**, not
> asserted. See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the literature + sources behind the recipe.

## The model & domain
- **Model:** `HuggingFaceTB/SmolLM2-135M` (base) and `SmolLM2-135M-Instruct` — Apache-2.0, tiny, fast.
- **Domain:** computational/quantum chemistry (DFT, Hartree–Fock, basis sets, coupled cluster, …) —
  dense technical vocab a tiny base model is weak on, so CPT shows a measurable effect.
- **Data:** CPT corpus from Wikipedia (CC BY-SA, attributed); SFT = a small, hand/synthesized Q&A set.

## Part A — Train (GPU-optional on the HF/PEFT path)
1. `01_build_corpus.py` — collect + clean Wikipedia comp/quantum-chem pages → raw CPT corpus.
2. `02_data_availability.py` — quantify the asymmetry: abundant raw text vs. scarce instruction pairs.
3. `03_cpt.py` — continued pretraining (LoRA incl. `embed_tokens`/`lm_head`); track held-out perplexity.
4. `04_cpt_base_vs_instruct.py` — CPT from base vs instruct; forgetting check.
5. `05_sft.py` — SFT the CPT'd model on the small Q&A set.
6. `06_base_vs_instruct_sweep.py` — **centerpiece**: {base,instruct} × N∈{10,30,100}, eval-vs-N curve.
7. `07_eval.py` — perplexity, held-out loss, side-by-side generations, simple rubric.
8. `08_unsloth_vs_hf.py` — same SFT both ways; compare code/speed/VRAM (needs GPU + isolated env).

## Part B — Deploy at the Edge (advanced)
9.  `09_merge_and_gguf.py` — merge LoRA → F16 GGUF → `llama-quantize` Q4_K_M.
10. `10_ollama_deploy.py` — Modelfile + `ollama create`; call the OpenAI-compatible API.
11. `11_edge_benchmark.md` + script — run on a Raspberry Pi-class device; measure tok/s + RAM.
12. `12_harness.py` — evaluate via **lm-evaluation-harness**; minimal agent/tool-use loop ("run as agent").

## Reproducibility
- All knobs (model IDs, seeds, paths, domain page list, hyperparams) live in `config.py`.
- Seeds set for torch/numpy/random + `transformers.set_seed`. Versions recorded at run time.
- Intermediate artifacts cached under `data/` and `outputs/` (gitignored where large).
- Scripts degrade gracefully without GPU/internet where possible.

## Layout
```
case_study/
  config.py            central config (edit here)
  01..12_*.py          pipeline steps
  data/                corpus + datasets (cached)
  outputs/             adapters, merged/GGUF models, metrics, plots
  RESEARCH_NOTES.md    literature + sources
  README.md            this file
```
Outputs feed back into the ebook: a new "Case Study" chapter in `../docs/handbook.html` + a notebook in `../notebooks/`.
