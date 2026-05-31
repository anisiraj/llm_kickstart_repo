"""
utils.py — shared helpers for the case study (importable normally; no digit prefix).

generate_samples() prints the model's answers to a few held-out questions for quick VISUAL
assessment at the end of a run. Greedy decoding (do_sample=False) keeps it deterministic and
reproducible. Used by §3 (CPT, completion-style) and reused by §5/§7 (SFT, chat-style).
"""
from __future__ import annotations

import torch


def generate_samples(model, tokenizer, questions: list[str] | None = None, *,
                     max_new_tokens: int = 64, chat: bool = False, n: int | None = None,
                     title: str = "sample generations") -> list[dict]:
    """Generate (greedy) answers to `questions` and print them. Returns the Q/A pairs.

    chat=False: feed the question text and let the model continue (right for a base/CPT model).
    chat=True : apply the tokenizer's chat template (right for an instruct/SFT model).
    """
    import config
    if questions is None:
        questions = config.SAMPLE_QUESTIONS
    if n is not None:
        questions = questions[:n]

    model.eval()
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    print(f"\n--- {title} (greedy; eyeball these) ---")
    results = []
    for q in questions:
        if chat and getattr(tokenizer, "chat_template", None):
            enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(device)
        else:
            ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=eos)
        answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({"question": q, "answer": answer})
        print(f"\nQ: {q}\nA: {answer}")
    print("")
    return results
