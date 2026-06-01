r"""
17_push_to_hf.py — publish the case-study artifacts to the Hugging Face Hub.

Pushes, with real model cards that cross-reference each other and the handbook:
  • dataset  anisiraj/comp-chem-quantum-chem        (corpus *.txt + seed_qa.jsonl)
  • adapter  anisiraj/SmolLM3-3B-compchem-sft-lora  (SFT LoRA, completion-only)
  • adapter  anisiraj/MiniCPM5-1B-compchem-sft-lora (SFT LoRA, completion-only)
  • adapter  anisiraj/SmolLM3-3B-compchem-cpt-lora  (CPT LoRA, +embed_tokens/lm_head)
  • model    anisiraj/MiniCPM5-1B-compchem-merged   (merged FP16, Ollama-ready)

The 12 GB SmolLM3 merged model is intentionally NOT pushed — it's derivable from base + SFT adapter.

Idempotent (create_repo exist_ok=True; uploads overwrite). Needs HF_TOKEN (or `huggingface-cli login`).
Run:  ../.venv/bin/python case_study/scripts/17_push_to_hf.py
      ../.venv/bin/python case_study/scripts/17_push_to_hf.py --dry-run   # list what WOULD push
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parent.parent          # case_study/
OUT = ROOT / "outputs"
DATA = ROOT / "data"
USER = "anisiraj"
DATASET_REPO = f"{USER}/comp-chem-quantum-chem"

# files we keep out of adapter repos (optimizer state / duplicate checkpoints are large + useless to others)
ADAPTER_ALLOW = ["adapter_config.json", "adapter_model.safetensors", "tokenizer*.json",
                 "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "*.md"]

HANDBOOK = "https://github.com/anisiraj/llm_kickstart_repo (handbook.html → 🧪 Case Study)"


def card(kind: str, base: str, phase: str, metrics: str) -> str:
    """A minimal-but-real model/dataset card with YAML frontmatter."""
    if kind == "dataset":
        return f"""---
license: cc-by-sa-4.0
language: [en]
tags: [chemistry, quantum-chemistry, computational-chemistry, continued-pretraining, sft]
pretty_name: Computational & Quantum Chemistry — CPT corpus + SFT seed
---

# Computational & Quantum Chemistry — fine-tuning corpus + SFT seed

Neutral, reproducible data for the CPT→SFT case study in {HANDBOOK}.

- `corpus/*.txt` — 41 deduplicated Wikipedia articles (comp/quantum chemistry), NFKC-normalized,
  boilerplate stripped, **equations preserved as inline `$LaTeX$`** (1,149 equations). ~112k tokens.
  Source: English Wikipedia, **CC BY-SA 4.0** (attribution required).
- `sft/seed_qa.jsonl` — 32 hand-written `{{prompt, completion}}` pairs for completion-only SFT.

Built by `case_study/scripts/01_build_corpus.py` + `02_data_availability.py`. The ~99× token gap
between corpus and Q&A is the whole point of the case study (instruction data is scarce).
"""
    return f"""---
license: apache-2.0
base_model: {base}
library_name: peft
tags: [lora, {phase}, chemistry, quantum-chemistry, rslora]
datasets: [{DATASET_REPO}]
---

# {base.split('/')[-1]} — computational-chemistry {phase.upper()} ({"merged" if kind=="model" else "LoRA adapter"})

{"Merged FP16 weights (base + SFT adapter), ready for GGUF/Ollama." if kind=="model" else
 f"A {'rsLoRA' } adapter from the CPT→SFT case study in {HANDBOOK}."}

- **Phase:** {phase} {"(full causal loss on raw domain text; targets include embed_tokens + lm_head)" if phase=="cpt" else "(completion-only loss on Q&A; prompt tokens masked to -100)"}
- **Base model:** `{base}`
- **Data:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})
- **Measured:** {metrics}
- **Reproduce:** `bash case_study/run.sh {"minicpm5" if "MiniCPM" in base else "smollm3"}`

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{USER}/{base.split('/')[-1]}-compchem-{phase}-lora")
tok = AutoTokenizer.from_pretrained("{base}")
```
""" if kind != "model" else f"""---
license: apache-2.0
base_model: {base}
tags: [chemistry, quantum-chemistry, sft, gguf-ready, ollama]
datasets: [{DATASET_REPO}]
---

# {base.split('/')[-1]} — computational-chemistry assistant (merged, Ollama-ready)

Merged FP16 weights = base `{base}` + the completion-only SFT adapter, from {HANDBOOK}.
Llama-architecture → Ollama imports it directly. **Measured:** {metrics}

```bash
# quantize to Q4_K_M (~0.7 GB) and serve locally
ollama create chem-minicpm5 -q q4_K_M -f Modelfile
ollama run chem-minicpm5 "What is the Hartree-Fock method?"
```
"""


JOBS = [
    # (kind, repo, local_path, base, phase, metrics, allow_patterns)
    ("dataset", DATASET_REPO, DATA, "", "", "", None),
    ("adapter", f"{USER}/SmolLM3-3B-compchem-sft-lora", OUT/"smollm3/sft_instruct_full/adapter",
     "HuggingFaceTB/SmolLM3-3B", "sft", "completion ppl 39.23→6.64 (-83%), recall 18→22%; unmasked 81%", ADAPTER_ALLOW),
    ("adapter", f"{USER}/MiniCPM5-1B-compchem-sft-lora", OUT/"minicpm5/sft_instruct_full/adapter",
     "openbmb/MiniCPM5-1B-sft", "sft", "completion ppl 107.68→3.64 (-97%), recall 5→46%; unmasked 78%", ADAPTER_ALLOW),
    ("adapter", f"{USER}/SmolLM3-3B-compchem-cpt-lora", OUT/"smollm3/cpt_base_full_unsloth/adapter",
     "HuggingFaceTB/SmolLM3-3B-Base", "cpt", "domain ppl 11.19→11.15; no forgetting (general 8.33→8.19)", ADAPTER_ALLOW),
    ("model", f"{USER}/MiniCPM5-1B-compchem-merged", OUT/"minicpm5/merged_full",
     "openbmb/MiniCPM5-1B-sft", "sft", "0.689 GB at Q4_K_M; completion ppl →3.64, recall 46%", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    print(f"HF user: {whoami()['name']}")
    for kind, repo, path, base, phase, metrics, allow in JOBS:
        rt = "dataset" if kind == "dataset" else "model"
        if not Path(path).exists():
            print(f"  [skip] {repo}: missing {path}")
            continue
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        print(f"\n{'='*60}\n{kind.upper():8} → {repo}  ({rt}, ~{size:.0f} MB from {path})")
        if a.dry_run:
            print("  [dry-run] would create repo + upload + write card"); continue
        api.create_repo(repo, repo_type=rt, exist_ok=True, private=False)
        (Path(path)/"README.md").write_text(card(kind, base, phase, metrics))
        api.upload_folder(repo_id=repo, repo_type=rt, folder_path=str(path),
                          allow_patterns=allow, commit_message="Add case-study artifact")
        print(f"  ✓ https://huggingface.co/{'datasets/' if rt=='dataset' else ''}{repo}")
    print("\nDone.")


if __name__ == "__main__":
    main()
