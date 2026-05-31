"""
config.py — single source of truth for the case study (edit here, not in the scripts).

Keeping every knob in one place is what makes the pipeline reproducible: same seed, same
model IDs, same page list, same hyperparameters → same results (within hardware variance).
"""
from __future__ import annotations
import os
import random
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CORPUS_DIR = DATA / "corpus"          # raw Wikipedia text (one file per page)
SFT_DIR = DATA / "sft"                # instruction Q&A sets
for _d in (DATA, OUTPUTS, CORPUS_DIR, SFT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Models (Apache-2.0, tiny) ─────────────────────────────────────────────────
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# ── Run mode: TRIAL (fast smoke test) vs FULL (the real run) ──────────────────
# Top-level flag every script/notebook reads. TRIAL caps data volume + training steps so the
# whole pipeline runs end-to-end in seconds (to validate syntax/shapes/plumbing); FULL does the
# real thing. Override per-process with env CASE_STUDY_MODE=full, or call set_mode("full").
import os as _os
RUN_MODE = _os.environ.get("CASE_STUDY_MODE", "trial").lower()


def set_mode(mode: str) -> None:
    """Set TRIAL/FULL at runtime (used by the notebooks' top-level flag cell)."""
    global RUN_MODE
    assert mode in ("trial", "full"), mode
    RUN_MODE = mode


def is_trial() -> bool:
    return RUN_MODE == "trial"


def limits() -> dict:
    """Mode-dependent volumes/steps. One place to tune both regimes."""
    if is_trial():
        return dict(cpt_char_cap=80_000, cpt_max_steps=8, sft_max_steps=8,
                    sweep_sizes=[10, 30], eval_blocks=2, gen_samples=2)
    return dict(cpt_char_cap=None, cpt_max_steps=-1, sft_max_steps=-1,
                sweep_sizes=SFT_SWEEP_SIZES, eval_blocks=12, gen_samples=8)


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def set_all_seeds(seed: int = SEED) -> None:
    """Seed python / numpy / torch (+ transformers if available) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed)
    except ImportError:
        pass


def record_env(path: Path | None = None) -> dict:
    """Capture library versions + device so every run is self-documenting."""
    info: dict[str, str] = {}
    import platform
    info["python"] = platform.python_version()
    for pkg in ("torch", "transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            import importlib.metadata as m
            info[pkg] = m.version(pkg)
        except Exception:
            info[pkg] = "—"
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["cuda"] = "n/a"
    if path:
        import json
        path.write_text(json.dumps(info, indent=2))
    return info


# ── Domain: computational & quantum chemistry Wikipedia pages (CPT corpus) ────
# Curated, neutral, factual. CC BY-SA — attribute in the chapter.
# NOTE: titles below were verified against the live API (2026-05-30). Known redirects that
# collapse onto another page were REMOVED so the corpus isn't silently duplicated:
#   "Self-consistent field method"→Hartree–Fock; "Hohenberg–Kohn theorems" &
#   "Generalized gradient approximation"→Density functional theory. Pages with no standalone
#   article ("Exchange–correlation functional", "Population analysis", "Psi4") were replaced.
# 01_build_corpus.py ALSO dedups by resolved title + content hash as a safety net.
WIKI_PAGES = [
    # quantum chemistry foundations
    "Quantum chemistry", "Schrödinger equation", "Born–Oppenheimer approximation",
    "Hartree–Fock method", "Slater determinant",
    "Molecular orbital", "Linear combination of atomic orbitals", "Basis set (chemistry)",
    "Gaussian orbital", "Electron correlation", "Configuration interaction",
    "Coupled cluster", "Møller–Plesset perturbation theory", "Multi-configurational self-consistent field",
    "Density matrix",
    # density functional theory
    "Density functional theory", "Kohn–Sham equations",
    "Local-density approximation", "Hybrid functional",
    # computational chemistry / methods & properties
    "Computational chemistry", "Molecular mechanics", "Force field (chemistry)",
    "Molecular dynamics", "Semi-empirical quantum chemistry method", "Ab initio quantum chemistry methods",
    "Potential energy surface", "Geometry optimization", "Normal mode",
    "Mulliken population analysis", "Partial charge", "Natural bond orbital",
    "Pseudopotential", "Plane wave", "Tight binding", "Quantum Monte Carlo",
    # software packages
    "Gaussian (software)", "NWChem", "PSI (computational chemistry)", "Q-Chem",
    "ORCA (quantum chemistry program)", "Quantum ESPRESSO",
]

# ── Hyperparameters (kept small + explicit) ───────────────────────────────────
CPT = dict(max_seq_len=1024, lr=5e-5, epochs=1, batch_size=4, grad_accum=4,
           # embeddings/lm_head adapt to the new-domain (math-heavy) distribution, but with a
           # SMALLER LR than the rest of the adapter to avoid destabilizing them (see PITFALLS.md).
           embedding_learning_rate=5e-6,
           lora_r=16, lora_alpha=16, lora_dropout=0.0,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj",
                           "embed_tokens", "lm_head"])
SFT = dict(max_seq_len=1024, lr=2e-4, epochs=3, batch_size=4, grad_accum=2,
           lora_r=16, lora_alpha=16, lora_dropout=0.0,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"])
# A few held-out questions printed at the end of every run for quick *visual* assessment.
# Deliberately overlap the domain so you can eyeball whether CPT/SFT changed the answers.
SAMPLE_QUESTIONS = [
    "What is the Hartree-Fock method?",
    "What does density functional theory compute?",
    "What is a basis set in quantum chemistry?",
    "What is electron correlation?",
    "What is coupled cluster theory?",
    "What is the Born-Oppenheimer approximation?",
]

SFT_SWEEP_SIZES = [10, 30, 100]   # centerpiece data-availability experiment
REPLAY_FRACTION = 0.10            # general-data rehearsal to mitigate forgetting
QUANT = "Q4_K_M"                  # GGUF quantization target

if __name__ == "__main__":
    import json
    print("Env:", json.dumps(record_env(), indent=2))
    print(f"\n{len(WIKI_PAGES)} Wikipedia pages queued for the CPT corpus.")
