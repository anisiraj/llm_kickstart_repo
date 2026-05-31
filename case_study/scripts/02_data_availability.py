r"""
02_data_availability.py — make research question #2 concrete: SFT data is the bottleneck.

THE POINT
---------
CPT and SFT need fundamentally different data, and they are NOT equally easy to get:
  • CPT wants RAW TEXT — abundant. §1 scraped ~100k tokens of clean domain text in seconds, no labels.
  • SFT wants INSTRUCTION PAIRS (prompt→completion) — scarce. Someone has to *author* correct answers.
    Below is a hand-written seed set for our domain. Notice how few there are: writing accurate Q&A is
    slow expert work, which is exactly why small-data SFT is the realistic regime (and why §6 asks
    whether to start from a base or an instruct model when data is tiny).

This script quantifies the asymmetry and shows the two DATA SHAPES, which map directly to two LOSSES:
  • corpus  -> {"text": ...}                 -> CPT, FULL causal loss on all tokens
  • SFT set -> {"prompt":..., "completion":...} -> SFT, loss on the COMPLETION only (prompt masked)
(See RESEARCH_NOTES.md §7 and PITFALLS.md.)

Run: python case_study/02_data_availability.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ── The scarce side: a hand-authored seed of instruction pairs ────────────────
# Neutral, factual computational/quantum-chemistry Q&A. This is the *expensive* data: every answer
# was written and checked by hand. ~30 pairs is already real effort — that is the lesson.
SEED_QA: list[dict[str, str]] = [
    {"prompt": "What does the Hartree–Fock method approximate?",
     "completion": "It approximates the many-electron wavefunction as a single Slater determinant and treats electron–electron repulsion in a mean-field (averaged) way, so it omits instantaneous electron correlation."},
    {"prompt": "What is a basis set in quantum chemistry?",
     "completion": "A set of functions—often Gaussian-type orbitals—used to expand molecular orbitals as linear combinations. Larger basis sets are more flexible and accurate but more expensive."},
    {"prompt": "What is the central idea of density functional theory?",
     "completion": "DFT reformulates the electronic problem in terms of the electron density instead of the many-body wavefunction; the Hohenberg–Kohn theorems show the ground-state energy is a functional of that density."},
    {"prompt": "What does the Born–Oppenheimer approximation assume?",
     "completion": "That nuclei are much heavier and move much more slowly than electrons, so nuclear and electronic motion can be separated and the electrons solved in a fixed field of nuclei."},
    {"prompt": "Define electron correlation.",
     "completion": "The energy difference between the exact non-relativistic energy and the Hartree–Fock energy, arising from the correlated instantaneous motion of electrons that mean-field theory misses."},
    {"prompt": "What is a Slater determinant?",
     "completion": "An antisymmetrized product of one-electron spin-orbitals; it satisfies the Pauli principle because exchanging two electrons changes its sign."},
    {"prompt": "What does the LCAO approximation do?",
     "completion": "It constructs molecular orbitals as linear combinations of atomic orbitals."},
    {"prompt": "What is coupled cluster theory?",
     "completion": "A post-Hartree–Fock method that captures electron correlation with an exponential cluster operator acting on a reference determinant; CCSD(T) is often called the gold standard of quantum chemistry."},
    {"prompt": "What role does the exchange–correlation functional play in DFT?",
     "completion": "It approximates the unknown exchange and correlation energy as a functional of the density; its choice (LDA, GGA, hybrid) largely determines a DFT calculation's accuracy."},
    {"prompt": "What is a pseudopotential?",
     "completion": "An effective potential that replaces core electrons so only valence electrons are treated explicitly, reducing cost and capturing core/relativistic effects."},
    {"prompt": "What correction does MP2 add to Hartree–Fock?",
     "completion": "Møller–Plesset second-order perturbation theory adds a perturbative estimate of electron correlation on top of the Hartree–Fock reference."},
    {"prompt": "What is a potential energy surface?",
     "completion": "The energy of a molecular system as a function of its nuclear coordinates; minima are stable structures and first-order saddle points are transition states."},
    {"prompt": "What is geometry optimization?",
     "completion": "Finding nuclear coordinates that make the energy stationary (zero forces) on the potential energy surface, typically a local minimum."},
    {"prompt": "What are normal modes of a molecule?",
     "completion": "The independent harmonic vibrations obtained by diagonalizing the mass-weighted Hessian; their frequencies underlie IR and Raman spectra."},
    {"prompt": "What is molecular dynamics?",
     "completion": "Simulating the time evolution of atoms by numerically integrating Newton's equations of motion using forces from a force field or a quantum method."},
    {"prompt": "What is a force field in molecular mechanics?",
     "completion": "A parameterized classical potential—bonds, angles, torsions, electrostatics, van der Waals—used to compute energies and forces without solving the electronic structure."},
    {"prompt": "What do the Kohn–Sham equations introduce?",
     "completion": "A fictitious system of non-interacting electrons that reproduces the true density, turning DFT into solvable single-particle equations with an exchange–correlation potential."},
    {"prompt": "What is a hybrid functional?",
     "completion": "A DFT functional that mixes a fraction of exact Hartree–Fock exchange with (semi)local exchange–correlation, e.g. B3LYP or PBE0."},
    {"prompt": "What is the local-density approximation?",
     "completion": "An exchange–correlation approximation that uses the density at each point as if it were a uniform electron gas."},
    {"prompt": "What is Mulliken population analysis?",
     "completion": "A scheme that partitions electrons among atoms using basis-function overlap populations to estimate atomic charges; it is basis-set dependent."},
    {"prompt": "What is a plane-wave basis set?",
     "completion": "A basis of plane waves, used with periodic boundary conditions in solid-state calculations and controlled by a kinetic-energy cutoff."},
    {"prompt": "What is configuration interaction?",
     "completion": "A post-Hartree–Fock method that expands the wavefunction as a linear combination of Slater determinants (excitations from a reference) to recover correlation."},
    {"prompt": "What is the self-consistent field procedure?",
     "completion": "Iteratively solving the Hartree–Fock or Kohn–Sham equations until the orbitals and the field they generate stop changing between iterations."},
    {"prompt": "How do ab initio and semi-empirical methods differ?",
     "completion": "Ab initio methods solve the electronic structure from first principles, while semi-empirical methods approximate or parameterize integrals using experimental data to lower cost."},
    {"prompt": "What is a Gaussian-type orbital?",
     "completion": "A basis function with a Gaussian radial part (e^{-αr²}); products of Gaussians integrate easily, which is why they are standard in molecular quantum chemistry."},
    {"prompt": "What is the tight-binding method?",
     "completion": "An approximate electronic-structure approach that expands wavefunctions in a small atomic-orbital basis with parameterized hopping integrals."},
    {"prompt": "What is the one-electron density matrix?",
     "completion": "An operator encoding orbital occupations and the electronic state; it lets many properties be computed without explicit wavefunctions."},
    {"prompt": "What is a partial charge?",
     "completion": "An assigned, generally non-integer atomic charge that approximates how electrons are distributed among the atoms of a molecule."},
    {"prompt": "What is MCSCF?",
     "completion": "Multi-configurational self-consistent field: it optimizes both the orbitals and the coefficients of several determinants, important when static correlation is strong."},
    {"prompt": "Name several computational chemistry software packages.",
     "completion": "Common examples include Gaussian, NWChem, PSI4, Q-Chem, ORCA, and Quantum ESPRESSO."},
    {"prompt": "Why is CCSD(T) called the gold standard?",
     "completion": "Because for many single-reference molecules it gives near-chemical-accuracy energies, so other methods are often benchmarked against it—at a steep computational cost."},
    {"prompt": "What problem does continued pretraining on domain text solve?",
     "completion": "It exposes a general model to a specialized corpus so it absorbs domain vocabulary and patterns, improving downstream performance before any instruction tuning."},
]


def build_sft_seed() -> Path:
    """Write the hand-authored seed Q&A to JSONL (prompt-completion shape). Returns the path."""
    config.set_all_seeds()
    out = config.SFT_DIR / "seed_qa.jsonl"
    with out.open("w") as f:
        for row in SEED_QA:
            f.write(json.dumps(row) + "\n")
    return out


def _est_tokens(words: int) -> int:
    return words * 4 // 3   # rough words->tokens estimate (keeps §2 offline/fast)


def analyze() -> dict:
    """Quantify the abundant (corpus) vs scarce (SFT) sides and return the stats."""
    seed_path = build_sft_seed()

    # abundant side: read §1's manifest
    man = json.loads((config.CORPUS_DIR / "manifest.json").read_text())
    corpus_words = sum(p["words"] for p in man["pages"])
    corpus_tokens = _est_tokens(corpus_words)

    # scarce side
    sft_words = sum(len((r["prompt"] + " " + r["completion"]).split()) for r in SEED_QA)
    sft_tokens = _est_tokens(sft_words)

    ratio = corpus_tokens / max(sft_tokens, 1)
    stats = dict(corpus_pages=len(man["pages"]), corpus_words=corpus_words, corpus_tokens=corpus_tokens,
                 sft_pairs=len(SEED_QA), sft_words=sft_words, sft_tokens=sft_tokens, ratio=ratio,
                 seed_path=str(seed_path))

    print("=== Data availability: abundant CPT text vs scarce SFT pairs ===\n")
    print(f"  CPT corpus  (raw text, shape {{'text': ...}})      : "
          f"{stats['corpus_pages']:>4} pages | ~{corpus_tokens:>7,} tokens  -> FULL causal loss")
    print(f"  SFT seed    (prompt/completion pairs)             : "
          f"{stats['sft_pairs']:>4} pairs | ~{sft_tokens:>7,} tokens  -> COMPLETION-only loss")
    print(f"\n  Asymmetry: the raw corpus has ~{ratio:.0f}× more tokens than the hand-built SFT set,")
    print(f"  and the corpus took *seconds* to collect while every Q&A answer was written by hand.")
    print(f"\n  Seed SFT written to {seed_path}")
    print("\n  Shapes drive losses:")
    print("    corpus -> {'text': ...}                  -> predict every token (CPT)")
    print("    sft    -> {'prompt':..., 'completion':...} -> predict the completion only (SFT)")
    print("\n  This scarcity is why §6 asks: with so few SFT pairs, start from BASE or INSTRUCT?")
    return stats


if __name__ == "__main__":
    analyze()
