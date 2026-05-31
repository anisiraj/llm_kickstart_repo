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
# config.py lives in case_study/scripts/, so the case-study root (where data/outputs live) is parent.parent.
ROOT = Path(__file__).resolve().parent.parent
import os as _os

DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"          # raw Wikipedia text (shared across models)
SFT_DIR = DATA / "sft"                # instruction Q&A sets (shared across models)

# ── Model registry — the pipeline runs on either model ────────────────────────
# Pick with env CASE_STUDY_MODEL or set_model(). SmolLM3-3B uses QLoRA (4-bit) so it fits a 12GB GPU;
# it must run in the Unsloth env (.venv) which has bitsandbytes. The 135M runs bf16 in either env.
MODELS = {
    "smollm2-135m": dict(base="HuggingFaceTB/SmolLM2-135M",
                         instruct="HuggingFaceTB/SmolLM2-135M-Instruct", load_in_4bit=False),
    "smollm3":      dict(base="HuggingFaceTB/SmolLM3-3B-Base",
                         instruct="HuggingFaceTB/SmolLM3-3B", load_in_4bit=True),
    # cheap QLoRA-path smoke test: the 135M loaded in 4-bit (verifies the SmolLM3 code path fast).
    "smollm2-135m-4bit": dict(base="HuggingFaceTB/SmolLM2-135M",
                              instruct="HuggingFaceTB/SmolLM2-135M-Instruct", load_in_4bit=True),
}
MODEL_KEY = _os.environ.get("CASE_STUDY_MODEL", "smollm2-135m").lower()

# §13: Ollama name for the stock tool-capable model (override with --model).
# Ollama's library has no `smollm3`; pull the official GGUF from HF instead:
#   ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
SMOLLM3_OLLAMA = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"

# Resolved per active model (updated by set_model). OUTPUTS is model-scoped so the two models'
# adapters/metrics never collide; DATA (corpus, seed Q&A) is shared.
BASE_MODEL = INSTRUCT_MODEL = LOAD_IN_4BIT = OUTPUTS = None  # set by _apply_model() below


def _apply_model() -> None:
    global BASE_MODEL, INSTRUCT_MODEL, LOAD_IN_4BIT, OUTPUTS
    m = MODELS[MODEL_KEY]
    BASE_MODEL, INSTRUCT_MODEL, LOAD_IN_4BIT = m["base"], m["instruct"], m["load_in_4bit"]
    OUTPUTS = ROOT / "outputs" / MODEL_KEY
    for _d in (DATA, CORPUS_DIR, SFT_DIR, OUTPUTS):
        _d.mkdir(parents=True, exist_ok=True)


def set_model(key: str) -> None:
    """Switch the active model (e.g. 'smollm2-135m' or 'smollm3') at runtime."""
    global MODEL_KEY
    assert key in MODELS, f"{key} not in {list(MODELS)}"
    MODEL_KEY = key
    _apply_model()


_apply_model()

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
                    sweep_sizes=[4, 8], eval_blocks=2, gen_samples=2)
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

# §4 catastrophic-forgetting probes: NON-chemistry text + instructions. If CPT on the domain
# corpus damages general ability, general-text perplexity rises and these answers degrade.
GENERAL_PAGES = ["Coffee", "Association football", "Piano", "Mount Everest"]
GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "List three primary colors.",
    "Write one short sentence about a dog.",
]

# §6 centerpiece: SFT-set sizes for the base-vs-instruct sweep. Capped by how many pairs we
# actually hand-authored (32) minus the held-out test set — which is itself the scarcity lesson.
SFT_TEST_HOLDOUT = 8              # fixed held-out Q&A for evaluating every sweep point
SFT_SWEEP_SIZES = [4, 8, 16, 24]  # train-set sizes (FULL); TRIAL uses a subset via limits()
REPLAY_FRACTION = 0.10            # general-data rehearsal to mitigate forgetting
QUANT = "Q4_K_M"                  # GGUF quantization target

# Memory profile: a 3B model in 4-bit on a 12GB GPU needs smaller batch + sequence + gradient
# checkpointing (the 135M is comfortable at the defaults above). Applied at import for the run.sh
# path (CASE_STUDY_MODEL set before import). GRAD_CKPT flags the manual-Trainer sections (§3-hf/§4).
GRAD_CKPT = bool(LOAD_IN_4BIT)
if LOAD_IN_4BIT:
    CPT.update(batch_size=1, grad_accum=8, max_seq_len=512)
    SFT.update(batch_size=1, grad_accum=4, max_seq_len=512)

if __name__ == "__main__":
    import json
    print("Env:", json.dumps(record_env(), indent=2))
    print(f"\n{len(WIKI_PAGES)} Wikipedia pages queued for the CPT corpus.")
