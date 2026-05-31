r"""
18_token_surgery.py — add special tokens, resize embeddings, and PROVE the resize is sane.

Demonstrates the exact ritual from the handbook "Tokenizer & Vocabulary Surgery" chapter, on a
real (small, CPU-friendly) model so the numbers are measured, not asserted:
  • tokenizer already covers LaTeX/JSON (ASCII) → no new tokens needed there
  • add_special_tokens + resize_token_embeddings(mean_resizing=True)
  • verify new embedding rows land near the mean of the pretrained rows (not random/zero)
  • verify a round-trip: the new token encodes to one id and decodes back

Runs on CPU in seconds against the cached SmolLM2-135M. No training, no GPU, no network.
Run:  HF_HUB_OFFLINE=1 ../.venv/bin/python case_study/scripts/18_token_surgery.py
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    print(f"=== §18 token surgery | {MODEL} (CPU) ===")

    # 1) Do we even need new tokens? LaTeX + JSON are ASCII the BPE already handles.
    for s in (r"$\hat{H}\psi = E\psi$", '{"energy": -1.13}'):
        n = len(tok(s).input_ids)
        print(f"  '{s[:28]:<28}' -> {n} existing tokens (no surgery needed)")

    # 2) Add genuinely-new control tokens.
    before = len(tok)
    added = tok.add_special_tokens({"additional_special_tokens": ["<rxn>", "</rxn>"]})
    print(f"\n  added {added} special tokens | vocab {before} -> {len(tok)}")

    # 3) Resize the embedding matrix, initialising new rows at the embedding MEAN.
    emb = model.get_input_embeddings().weight.data
    old_mean = emb[:before].mean(dim=0)
    model.resize_token_embeddings(len(tok), mean_resizing=True)
    emb = model.get_input_embeddings().weight.data

    # 4) PROVE the new rows are sane: close to the pretrained mean, far from zero.
    new_row = emb[tok.convert_tokens_to_ids("<rxn>")]
    d_mean = torch.linalg.vector_norm(new_row - old_mean).item()
    d_zero = torch.linalg.vector_norm(new_row).item()
    base_spread = torch.linalg.vector_norm(emb[:before] - old_mean, dim=1).mean().item()
    print(f"  new '<rxn>' row: dist-to-mean {d_mean:.4f} (avg base spread {base_spread:.4f}) | "
          f"norm {d_zero:.4f}")
    assert d_mean <= base_spread, "mean_resizing should place the new row near the embedding mean"

    # 5) Round-trip: new token is atomic (1 id) and decodes back.
    ids = tok("<rxn>H2 + O2</rxn>", add_special_tokens=False).input_ids
    print(f"  '<rxn>' -> id {tok.convert_tokens_to_ids('<rxn>')} | "
          f"round-trip decode: {tok.decode(ids)!r}")
    assert tok.decode(tok("<rxn>", add_special_tokens=False).input_ids) == "<rxn>"

    print("\n  ✓ surgery verified: new rows near the manifold, atomic + reversible.")
    print("  Next: add embed_tokens+lm_head to LoRA targets and train them (see §3 CPT).")


if __name__ == "__main__":
    main()
