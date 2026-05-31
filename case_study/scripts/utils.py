"""
utils.py — shared helpers for the case study (importable normally; no digit prefix).

generate_samples() prints the model's answers to a few held-out questions for quick VISUAL
assessment at the end of a run. Greedy decoding (do_sample=False) keeps it deterministic and
reproducible. Used by §3 (CPT, completion-style) and reused by §5/§7 (SFT, chat-style).
"""
from __future__ import annotations

import torch


def load_causal_lm(name: str, *, training: bool = False, dtype=None):
    """Load a causal LM honoring the active model's precision.

    - config.LOAD_IN_4BIT (SmolLM3): QLoRA 4-bit via bitsandbytes (NF4 + bf16 compute, double-quant),
      placed by device_map; if training, prepare_model_for_kbit_training. Requires the .venv env.
    - otherwise (135M): bf16 on CUDA / fp32 on CPU, moved to the device.
    Returns the model already on its device, so callers must NOT call .to(...) on a 4-bit model.
    """
    import config
    from transformers import AutoModelForCausalLM
    cuda = torch.cuda.is_available()
    if config.LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        # keep non-quantized layers (norms, lm_head, embeddings) in bf16 so they match the bf16
        # autocast trl uses — otherwise SFTTrainer hits "expected BFloat16 but found Float".
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=qc, device_map="auto", dtype=torch.bfloat16)
        if training:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype or (torch.bfloat16 if cuda else torch.float32))
    return model.to("cuda") if cuda else model


def build_sft_model(model_name: str):
    """Return (peft_model, tokenizer) ready for SFTTrainer (pass WITHOUT peft_config).

    - 4-bit (SmolLM3): use Unsloth's FastLanguageModel — it's purpose-built for QLoRA and handles all
      the dtype/grad-checkpoint alignment that trips up a hand-rolled transformers+bnb+trl QLoRA
      (the 'expected BFloat16 but found Float' error). Requires the .venv env.
    - bf16 (135M): standard transformers + PEFT, pre-wrapped.
    """
    import config
    cfg = config.SFT
    if config.LOAD_IN_4BIT:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=cfg["max_seq_len"], dtype=None, load_in_4bit=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = FastLanguageModel.get_peft_model(
            model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], use_rslora=cfg["use_rslora"],
            use_gradient_checkpointing="unsloth", random_state=config.SEED)
        return model, tok
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = get_peft_model(load_causal_lm(model_name), LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        use_rslora=cfg["use_rslora"], target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    return model, tok


import math
import re

_STOP = {"the", "a", "an", "of", "is", "are", "to", "and", "in", "for", "that", "with",
         "as", "by", "it", "on", "or", "be", "this", "from", "which", "its", "into"}


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3 and w not in _STOP}


def keyword_recall(gold: str, gen: str) -> float:
    """Fraction of the gold answer's content words that appear in the generation (format-robust)."""
    g = content_words(gold)
    return len(g & content_words(gen)) / len(g) if g else 0.0


@torch.no_grad()
def completion_perplexity(model, tok, pairs: list[dict], device) -> float:
    """Mean perplexity of the gold completions given their prompts (prompt tokens masked).

    Mirrors the SFT objective (loss on answer tokens only); same formatting for every condition.
    Shared by §6 and §7 (lives here to avoid a circular import between them).
    """
    model.eval()
    total_loss, total_tok = 0.0, 0
    for p in pairs:
        p_ids = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(" " + p["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + c_ids], device=device)
        labels = torch.tensor([[-100] * len(p_ids) + c_ids], device=device)
        loss = model(ids, labels=labels).loss.item()
        total_loss += loss * len(c_ids)
        total_tok += len(c_ids)
    return math.exp(total_loss / max(total_tok, 1))


def generate_samples(model, tokenizer, questions: list[str] | None = None, *,
                     max_new_tokens: int = 64, chat: bool = False, n: int | None = None,
                     title: str = "sample generations", quiet: bool = False) -> list[dict]:
    """Generate (greedy) answers to `questions` and (unless quiet) print them. Returns the Q/A pairs.

    chat=False: feed the question text and let the model continue (right for a base/CPT model).
    chat=True : apply the tokenizer's chat template (right for an instruct/SFT model).
    quiet=True: don't print (used when scoring the whole test set in §7).
    """
    import config
    if questions is None:
        questions = config.SAMPLE_QUESTIONS
    if n is not None:
        questions = questions[:n]

    model.eval()
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    if not quiet:
        print(f"\n--- {title} (greedy; eyeball these) ---")
    results = []
    for q in questions:
        if chat and getattr(tokenizer, "chat_template", None):
            enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        else:
            enc = tokenizer(q, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=eos)
        answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({"question": q, "answer": answer})
        if not quiet:
            print(f"\nQ: {q}\nA: {answer}")
    if not quiet:
        print("")
    return results
