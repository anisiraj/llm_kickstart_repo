r"""
20_merge_models.py — fuse two same-base models in weight space (no training), and prove it loads.

Implements the math behind mergekit on REAL tensors (cached SmolLM2-135M base vs -Instruct), so the
handbook "Model Merging" chapter shows measured behaviour, not hand-waving:
  • task vector  tau = instruct - base
  • Linear  : 0.5*base + 0.5*instruct          (== base + 0.5*tau)
  • TIES    : keep top-k% |tau|, then base + tau   (trim → reduces interference)
  • DARE    : randomly drop p of tau, rescale 1/(1-p), then base + tau
We report how much each method moves the weights, then SAVE the linear merge, reload it, and generate
to prove the merged checkpoint is valid. CPU, seconds, offline.

For production multi-model merges use mergekit (`mergekit-yaml merge.yaml ./out`); this script is the
"what it actually does" version.
Run:  HF_HUB_OFFLINE=1 ../.venv/bin/python case_study/scripts/20_merge_models.py
"""
from __future__ import annotations
import copy
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-135M"
INST = "HuggingFaceTB/SmolLM2-135M-Instruct"


def sd_norm(sd) -> float:
    return sum(float(v.float().pow(2).sum()) for v in sd.values()) ** 0.5


def main() -> None:
    torch.manual_seed(0)
    print(f"=== §20 weight-space merge | {BASE.split('/')[-1]} (CPU) ===")
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    inst = AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32)
    b, i = base.state_dict(), inst.state_dict()

    # task vector tau = instruct - base (only over shared float params)
    keys = [k for k in b if k in i and b[k].shape == i[k].shape and b[k].is_floating_point()]
    tau = {k: i[k] - b[k] for k in keys}
    print(f"  shared params: {len(keys)} | ||tau|| (instruct-base) = {sd_norm(tau):.2f}")

    def build(method: str):
        out = copy.deepcopy(b)
        for k in keys:
            t = tau[k]
            if method == "linear":
                out[k] = b[k] + 0.5 * t                              # average of the two models
            elif method == "ties":                                   # keep top-20% |delta|, drop rest
                a = t.abs().flatten()
                s = a if a.numel() <= 1_000_000 else a[torch.randperm(a.numel())[:1_000_000]]
                thr = s.quantile(0.80) if s.numel() > 1 else a        # subsample: quantile() caps ~16M
                out[k] = b[k] + torch.where(t.abs() >= thr, t, torch.zeros_like(t))
            elif method == "dare":                                   # drop 50%, rescale survivors
                mask = (torch.rand_like(t) > 0.5).float()
                out[k] = b[k] + (t * mask) / 0.5
        return out

    for m in ("linear", "ties", "dare"):
        merged = build(m)
        drift = sd_norm({k: merged[k] - b[k] for k in keys})
        print(f"  {m:7} merge: moved {drift:7.2f} from base "
              f"({100*drift/sd_norm(tau):4.0f}% of the full task vector)")

    # Save the linear merge, reload, and PROVE it's a valid generating model.
    base.load_state_dict(build("linear"))
    with tempfile.TemporaryDirectory() as d:
        base.save_pretrained(d); AutoTokenizer.from_pretrained(BASE).save_pretrained(d)
        m2 = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(d)
        ids = tok("The Hartree-Fock method", return_tensors="pt").input_ids
        out = m2.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        print(f"\n  ✓ linear-merged model reloaded + generated: "
              f"{tok.decode(out[0], skip_special_tokens=True)!r}")
    print("  Merging redistributes existing skills with zero training; it can't create new capability.")


if __name__ == "__main__":
    main()
