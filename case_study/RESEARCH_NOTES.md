# Research Notes — CPT→SFT→Edge Case Study

Compiled 2026-05-30. Sources at the bottom of each section. These notes ground the running examples; numbers we report in the chapter must be **measured on our hardware** (RTX 3080 Ti / target edge device), not copied from here.

## 1. CPT → SFT recipe (the common recipe)
- **What CPT is for:** steer a base model toward a new/out-of-distribution domain (law, medicine, a language — here: computational/quantum chemistry). Base models saw trillions of tokens but may be thin on niche domains. CPT adds domain knowledge before SFT teaches the response format.
- **Data shapes:** CPT = raw corpus (books/articles/Wikipedia). SFT = instruction (Alpaca: instruction/input/output) or conversation (ShareGPT: role-tagged turns).
- **CPT-specific LoRA tip:** include `embed_tokens` and `lm_head` in `target_modules` so the embedding layer can adapt to new domain tokens (Unsloth docs + the user's own guide §8 agree).
- **The killer combo:** CPT on domain text → then SFT on a small instruction set in the same domain.
- **Catastrophic forgetting mitigations:** (a) **rehearsal** — mix ~10% general-purpose data into the CPT corpus; (b) **lower learning rate** preserves base capabilities (higher LR → more forgetting); (c) LoRA (vs full FT) inherently limits drift.

## 2. base vs instruct init under SMALL data (the centerpiece question)
- Common practice: continue SFT from **instruct** checkpoints, not base.
- Research signal (Mapping Post-Training Forgetting; "Best Instruction-Tuning Data Are Those That Fit"): in **low-data regimes**, post-training from an **instruct** model yields **low forgetting** and works well; from a **base** model you must teach instruction-following from scratch, which needs much more data. In high-data regimes the advantage washes out.
- → This supports our hypothesis (guide's "Recipe E"): **with small SFT data, init from instruct.** Our 2×3 sweep ({base,instruct}×N∈{10,30,100}) should show instruct-init reaching usable behavior with far fewer examples.

## 3. Convert fine-tuned model → GGUF (llama.cpp)
- **Merge first:** `convert_hf_to_gguf.py` does NOT accept PEFT adapters — merge LoRA into base weights first (`model.merge_and_unload()` / `save_pretrained_merged`). (Alternative: `convert_lora_to_gguf.py` to ship the adapter separately + Ollama `ADAPTER`.)
- **Two-step:** always export **F16 GGUF first**, then `llama-quantize` to the target (e.g. **Q4_K_M** = ~¼ size, quality close to original; the sensible default).
- **Gotchas:** subtle `config.json` / `tokenizer_config.json` / chat-template mismatches silently produce malformed GGUF that crashes at inference — patch config before converting. Tiny models convert in well under a minute.

## 4. Deploy with Ollama
- Create a **Modelfile**: `FROM ./model.gguf` + `PARAMETER` (temperature, num_ctx, stop) + `SYSTEM` + `TEMPLATE` (chat template). Then `ollama create my-model -f Modelfile`.
- For adapters: `FROM <base>` + `ADAPTER ./adapter.gguf` (base must match the adapter's base).
- Ollama serves a REST API at `:11434` and an **OpenAI-compatible** `/v1/chat/completions` → works with OpenAI SDKs / Open WebUI unmodified.

## 5. Embedded / edge performance (why small models matter)
- RPi 5 (8GB): 1–3B models at **10–18 tok/s**; RPi 4: TinyLlama 1.1B Q4 ~**8–12 tok/s**. A **135M** model will be dramatically faster + tiny RAM → genuinely real-time on a Pi/phone/microcontroller-class board.
- **Llamafile** often beats Ollama on Pi (3–4× faster, lower power) by exploiting CPU heterogeneity. BitNet 1.58 extremely RAM-efficient.
- Lesson for the chapter: quantized small models are the *practical* edge story; we measure tok/s + RAM on our device and report honestly.

## 6. "Harness" — evaluation + agent
- **lm-evaluation-harness** (EleutherAI, `lm-eval`): the standard. Eval a GGUF either via **hf backend** (`--model hf --model_args pretrained=DIR,gguf_file=FILE,tokenizer=DIR`) or via **llama.cpp server** + API. ⚠️ always pass an explicit tokenizer — letting HF reconstruct it from GGUF can hang for hours.
- Tasks: standard (hellaswag etc.) for general capability regression, **plus a small custom domain task** (chemistry Q&A / perplexity) to show CPT/SFT effect.
- **Agent harness:** deployed model behind Ollama's OpenAI-compatible API → drive it with a minimal tool-use/agent loop ("run as an agent"). Small models are weak at this — report honestly what a 135M can/can't do.

## 7. ★ Loss masking — full causal loss vs. loss-on-generation-only (the key teaching point)
This is the single most valuable distinction in the guide. *When* you compute loss on which tokens
depends entirely on what you are training:

| Phase / data | Loss computed on | TRL switch | Why |
|---|---|---|---|
| **CPT / continued pretraining** (raw text, our Wikipedia corpus) | **ALL tokens** (full causal LM loss) | language-modeling dataset (`{"text": …}`); no prompt/completion split | The whole point is to absorb the domain *distribution* — every token is signal. Predict-the-next-token over the entire stream. |
| **SFT on prompt→completion** (our small Q&A set) | **completion only** (prompt masked) | prompt-completion dataset; `completion_only_loss=True` (the **default** for this format) | You don't want the model to "learn" to generate the *question*; only the *answer* (the generation) carries the supervised signal. "Loss only on generation." |
| **SFT on multi-turn chat** | **assistant turns only** | conversational dataset; `assistant_only_loss=True` | Same idea across turns — never compute loss on user/system text, only what the model should produce. Requires the chat template to mark generation spans with `{% generation %}…{% endgeneration %}`; TRL auto-patches known families (e.g. Qwen3). |

Mechanics (TRL/Transformers): labels are the input shifted by one; masked positions use ignore-index **−100** so they contribute zero gradient. Padding is always −100. A healthy run has a sane *fraction of unmasked tokens* (CPT ~100%; completion-only ~30–70%; assistant-only ~20–60%) — if "all labels are −100" the loss flatlines (the guide's debugging §13).

→ In our case study: **§3 CPT uses full causal loss** (raw corpus, `text` field); **§5/§6 SFT uses completion-only loss** (`completion_only_loss=True`, prompt masked). We will print the unmasked-token fraction in each to *prove* the masking is doing what we claim — not assert it.

## 8. Token handling: special tokens, EOS, and our math/LaTeX decision
- **Don't add special tokens you don't need.** HF: chat templates already include the needed special tokens; adding extra ones is "often incorrect or duplicated, hurting performance." When you `apply_chat_template(tokenize=False)` then tokenize, pass `add_special_tokens=False` to avoid duplicate BOS/EOS. Training: `add_generation_prompt=False`.
- **Adding genuinely new tokens** (e.g. `<THINKING>`, `<SCRATCH_PAD>`): Unsloth `add_new_tokens(model, tokenizer, new_tokens=[...])` **before** `get_peft_model()`; under the hood this requires `tokenizer.add_special_tokens` + `model.resize_token_embeddings(len(tokenizer))`. Only do this for new *control* tokens, not for content the existing vocab can already represent.
- **EOS alignment:** for base models that ship a chat template (Qwen-style), set `eos_token` in `SFTConfig` (e.g. `"<|im_end|>"`) so responses terminate correctly. Document the EOS we use for SmolLM2.
- **Our LaTeX/math decision (documented in `01_build_corpus.py`):** Wikipedia `explaintext` emits each equation twice — a broken glyph-by-glyph Unicode dump, then clean LaTeX inside `{\displaystyle …}`. We **drop the glyph soup** and **keep the LaTeX**, normalized into inline `$ … $`. We do **NOT** add new vocab tokens for math: LaTeX is ASCII (`\`, `{`, `}`, `^`, `_`, letters) and the existing BPE tokenizer represents it fine. CPT then adapts the *existing* embeddings to the heavier math/LaTeX distribution — which is exactly why CPT includes `embed_tokens` + `lm_head` in the LoRA `target_modules` (§1 research + guide §8). This keeps the pipeline reproducible (no embedding-resize step) and honest.

## 9. Observed HF-PEFT vs Unsloth CPT differences (measured, TRIAL, SmolLM2-135M)
Running §3 both ways surfaced concrete, documentable differences (numbers are trial-run, for *shape* not final results):
- **Trainable params:** HF/PEFT ≈ **6.5M (4.6%)** — it LoRA-adapts `embed_tokens`/`lm_head`. Unsloth ≈ **34M (17.3%)** — it **fully trains the embeddings in mixed precision** (logs "Training embed_tokens in mixed precision", offloads to save VRAM) rather than LoRA-adapting them. Different philosophy, same goal (adapt embeddings to new-domain tokens).
- **Embedding LR:** Unsloth honors `embedding_learning_rate` natively (logs "Setting lr = 5.00e-06 instead of 5.00e-05 for embed_tokens"). On the HF path we replicate this manually with two optimizer param-groups.
- **Packing:** HF path = manual concatenation packing (we cut docs into fixed blocks); Unsloth = `packing=True` and auto-enables **padding-free** training.
- **Tied embeddings gotcha (both backends):** SmolLM2 has `tie_word_embeddings=True`; PEFT warns when a tied layer is in the adapter and auto-sets `save_embedding_layers=True`. Matters for GGUF export (Part B).
- **Env:** HF path runs in `.venv-rl` (trl 1.2); Unsloth path requires `.venv` (unsloth 2026.4.8 + trl 0.24). Same script, `--backend` flag, separate output dirs so both coexist.

## 10. Validation: is "CPT → SFT" (and our model choice) a sound recipe? (literature)
**The recipe is well-supported.** CPT-then-SFT is the standard domain-adaptation pipeline:
- **"Reuse, Don't Retrain: A Recipe for Continued Pretraining"** (arXiv 2407.07263) — the canonical CPT recipe.
- **"Domain-Adaptive Continued Pre-Training of Small Language Models"** (arXiv 2504.09687) — directly about *small* models (our regime).
- **"Modelling the Optimal Trade-Off Between CPT and SFT for LLM Domain Adaptation"** (OpenReview guUUlHPXRw) — formalizes the CPT↔SFT split.
- **CPT on an *instruct* model** + model merging: "Domain Adaptation of Llama3-70B-**Instruct** through Continual Pre-Training and Model Merging" (arXiv 2406.14971) — supports CPT *on instruct* and recovering general ability via merging/replay.
- npj Computational Materials 2025 (s41524-025-01564-y) — training strategies, scaling, merging for domain adaptation.
- **Practical knobs the literature converges on** (we follow these): mix ~**1:1 domain:replay** to prevent forgetting (works even at 3B); **stop CPT after 1–2 passes** over in-domain data (overfitting); in data-constrained settings use **much higher weight decay** (up to ~30×); **replay + distillation** are the most effective forgetting mitigations. → our config (1 epoch, replay fraction, low LR) is in line; consider raising weight decay for FULL.

**Model choice is sound.** SmolLM2 (135M/360M/1.7B) is state-of-the-art for its size and explicitly suited to LoRA/QLoRA domain fine-tuning (SmolLM2 paper arXiv 2502.02737; it beats Qwen2.5 base on HellaSwag/ARC at size). Qwen2.5-0.5B-Instruct is a strong alternative tiny instruct model. For the tool-capable end, **SmolLM3-3B** is the right pick. (distil labs benchmarked 12 SLMs for fine-tuning base selection — useful if revisiting.) **SmolLM2-360M** is a sensible middle option if 135M underperforms.

**SFTTrainer specifics confirmed** (HF/TRL docs): `completion_only_loss=None` defaults to completion-only for prompt-completion data and full-sequence for LM data; adapter LR ≈ **1e-4** (vs 2e-5 full FT); `packing=True` for throughput; LoRA is the recommended path. (Matches our §5/§6.)

## Sources
- Unsloth — Continued Pretraining: https://unsloth.ai/docs/basics/continued-pretraining ; blog https://unsloth.ai/blog/contpretraining ; fine-tuning guide https://unsloth.ai/docs/get-started/fine-tuning-llms-guide ; notebooks https://unsloth.ai/docs/get-started/unsloth-notebooks
- "Reuse, Don't Retrain: A Recipe for Continued Pretraining" — https://arxiv.org/pdf/2407.07263
- Chris McCormick — Continuing Pre-Training on Raw Text: https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/
- Mapping Post-Training Forgetting at Scale — https://arxiv.org/pdf/2510.17776 ; Best Instruction-Tuning Data — https://arxiv.org/html/2502.04194v2
- GGUF after fine-tune — https://markaicode.com/gguf-quantization-after-fine-tuning-llama-cpp/ ; https://zenvanriel.com/ai-engineer-blog/gguf-export-after-lora-training-step-by-step/
- Ollama import — https://docs.ollama.com/import ; Modelfile guide https://localaimaster.com/blog/ollama-modelfile-guide
- RPi/edge benchmarks — https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5 ; SBC eval https://arxiv.org/html/2511.07425v1
- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness ; GGUF eval issue https://github.com/EleutherAI/lm-evaluation-harness/issues/2525
- HF chat templating (special tokens, add_generation_prompt, training) — https://huggingface.co/docs/transformers/main/en/chat_templating
- TRL SFTTrainer (full vs completion-only vs assistant-only loss, EOS, packing, label shift/−100) — https://huggingface.co/docs/trl/sft_trainer
- Unsloth chat templates / add_new_tokens — https://unsloth.ai/docs/basics/chat-templates
- Corpus prep: D4 dedup https://arxiv.org/pdf/2308.12284 ; Fewer Truncations Improve LM (best-fit packing) https://arxiv.org/pdf/2404.10830 ; HF sequence packing https://huggingface.co/blog/sirluk/llm-sequence-packing
- Recipe validation: Reuse Don't Retrain https://arxiv.org/pdf/2407.07263 ; Domain-Adaptive CPT of Small LMs https://arxiv.org/abs/2504.09687 ; CPT↔SFT trade-off https://openreview.net/forum?id=guUUlHPXRw ; CPT on Llama3-70B-Instruct + merging https://arxiv.org/html/2406.14971v1 ; npj Comp Materials https://www.nature.com/articles/s41524-025-01564-y
- Model choice: SmolLM2 paper https://arxiv.org/html/2502.02737v1 ; distil labs SLM fine-tuning benchmark https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/
- SmolLM3 GGUF (Ollama pull): hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
