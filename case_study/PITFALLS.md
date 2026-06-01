# Common Pitfalls & Tricks — CPT / SFT (the stuff that actually bites)

A field guide to the failure modes you *will* hit. Each entry: the symptom, the cause, the fix, and
how to **detect it with a runnable check** (in keeping with the project rule: prove, don't assert).
Neutral examples only. Sources at the bottom.

> The single most common root cause across all of these is a **loss-mask / chat-template mismatch**.
> When in doubt, inspect the labels (trick #1 below).

## Pitfalls

### 1. "All my labels are −100" → loss flatlines, zero gradient
The classic. Loss won't move because every target token is masked out (ignore-index −100).
**Detect (runnable):**
```python
ex = trainer.train_dataset[0]
unmasked = sum(l != -100 for l in ex["labels"]); total = len(ex["labels"])
print(f"unmasked: {unmasked}/{total} ({100*unmasked/total:.1f}%)")
```
**Healthy fractions:** full causal/CPT ≈ **100%**; `completion_only_loss=True` ≈ **30–70%**; `assistant_only_loss=True` ≈ **20–60%**. If **0%**, the chat template lacks the expected markers (for `assistant_only_loss` it needs `{% generation %}…{% endgeneration %}`; for the manual completion-only path the response template string must match your data exactly). This is the most-reported TRL/Unsloth issue (often surfaces as `ZeroDivisionError: All labels are -100`).

### 2. "Model echoes the user's prompt back"
**Cause:** forgot `completion_only_loss=True`, so loss was computed on the prompt too — the model learned to *produce* prompts. **Fix:** enable completion-only loss; retrain.

### 3. "Loss decreases but generation is gibberish"
**Cause:** trained a **base** model with a chat template but never added the template's special tokens — the model sees `<|im_start|>` as random characters. **Fix:** set `chat_template_path` in `SFTConfig` (it handles token addition + embedding resize), or add the tokens explicitly (`tokenizer.add_special_tokens` + `model.resize_token_embeddings`) and align `eos_token`.

### 4. "Model loses general capability after CPT" (catastrophic forgetting)
**Cause:** over-training (>1 epoch), LR too high, or no replay data. **Fix:** lower LR, ≤1 epoch, mix ~10% general "replay" text into the CPT corpus. **Detect:** run a fixed general-capability smoke-test *before* SFT; if it regresses, restart with lower LR + replay. (We measure this in §4.)

### 5. "Training is way slower than expected"
Check, in order: `packing` (**ON** for CPT raw text; often **OFF** for SFT with completion-only loss), `gradient_checkpointing` (ON — default in `SFTConfig`), `bf16=True` (default if supported), and — if you meant to use it — that `FastLanguageModel.from_pretrained` (Unsloth) is actually the model being trained.

## The 5 beginner traps (quick hits)
1. **Echoing the prompt** → forgot completion-only loss (see #2).
2. **"Chat template isn't applying!"** → dataset column is `text` instead of `messages`; rename it and TRL templates it.
3. **Gibberish after training a base model** → imported the template but not its special tokens; use `chat_template_path` (see #3).
4. **"Should I write my own DataCollator?"** → almost never. Trust the defaults.
5. **"Loss is low but generation is bad"** → inspect `trainer.train_dataset[0]["labels"]`: are user tokens −100? Is the assistant span unmasked? (see #1).

## Tricks worth internalizing
- **CPT learning rate ≈ 5e-5** (standard). SFT adapters use a higher LR (≈ 1e-4–2e-4).
- **`embedding_learning_rate`**: when you add `embed_tokens`/`lm_head` to LoRA `target_modules` (needed so embeddings adapt to a new-domain/heavy-math distribution), train them with a **smaller** LR than the rest (e.g. ~5e-6–1e-5). Unsloth exposes `embedding_learning_rate` for exactly this.
- **Packing**: ON for CPT (pack documents to fill the context, separated by EOS); consider OFF for completion-only SFT so masking stays clean.
- **EOS alignment**: base models that ship a chat template (Qwen-style) need `eos_token` set in `SFTConfig` so responses terminate.
- **Inspect the mask, always.** The label-fraction check (#1) catches the majority of "my training looked fine but the model is broken" cases in 3 lines.

## Sources
- TRL SFT Trainer — https://huggingface.co/docs/trl/main/en/sft_trainer ; Chat Templates — https://huggingface.co/docs/trl/chat_templates
- Unsloth Continued Pre-Training — https://unsloth.ai/docs/basics/continued-pretraining ; Troubleshooting/FAQs — https://docs.unsloth.ai/basics/troubleshooting-and-faqs
- Issues: TRL #4879 (auto `{% generation %}`) ; Unsloth #2734 ("all labels −100") ; Unsloth #2771 (`train_on_responses_only` w/ Qwen3)
- Forgetting / CPT: arXiv 2504.09687 (domain-adaptive CPT of small LMs) ; arXiv 2406.14971 (CPT + model merging) ; arXiv 2504.17780 (replay to remember)
