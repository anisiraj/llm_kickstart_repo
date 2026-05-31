r"""
19_synth_data.py — generate synthetic SFT pairs with a teacher model, then QC them.

The case study hit a ~99x gap between corpus tokens and hand-written Q&A. This script closes it
the cheap way (distillation): a TEACHER model writes (prompt, completion) pairs for our domain,
then we run the quality control that actually matters:
  • decontaminate against the held-out seed set (no eval leakage)
  • dedupe (normalized exact-match hash)
  • filter (length / empties)

Teacher = a local Ollama model (default the chem-tuned MiniCPM5 we deployed), so it runs offline and
free. Gracefully degrades: if Ollama isn't up it still demonstrates the QC pipeline on a fixed sample.
Run:  ../.venv/bin/python case_study/scripts/19_synth_data.py
      SYNTH_TEACHER=chem-sft-smollm3-full:latest ../.venv/bin/python case_study/scripts/19_synth_data.py
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "sft" / "seed_qa.jsonl"
OUT = ROOT / "data" / "sft" / "synth_qa.jsonl"
OLLAMA = "http://localhost:11434"
TEACHER = os.environ.get("SYNTH_TEACHER", "chem-sft-smollm3-full:latest")

TOPICS = ["the variational principle", "Koopmans' theorem", "the LCAO approximation",
          "Mulliken population analysis", "the self-consistent field cycle", "spin contamination"]


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3); return True
    except Exception:
        return False


def teacher_pair(topic: str) -> dict | None:
    """Ask the teacher for one Q&A about `topic`; parse to a (prompt, completion) pair."""
    prompt = (f"Write ONE short question about {topic} in computational chemistry, then a concise "
              f"2-sentence answer. Format exactly:\nQ: <question>\nA: <answer>")
    body = json.dumps({"model": TEACHER, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.7, "num_predict": 160}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(req, timeout=120)).get("response", "")
    except Exception as e:
        print(f"  [gen error] {e}"); return None
    q = re.search(r"Q:\s*(.+)", txt)
    a = re.search(r"A:\s*(.+)", txt, re.S)
    if not (q and a):
        return None
    return {"prompt": q.group(1).strip(), "completion": " ".join(a.group(1).split())}


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def qc(pairs: list[dict], seed: list[dict]) -> tuple[list[dict], dict]:
    """decontaminate vs seed, dedupe, length-filter. Returns (kept, stats)."""
    seed_keys = {norm(p["prompt"]) for p in seed}
    kept, seen, stats = [], set(), {"in": len(pairs), "leak": 0, "dup": 0, "short": 0}
    for p in pairs:
        k = norm(p["prompt"])
        if not p["prompt"] or len(p["completion"]) < 20:
            stats["short"] += 1; continue
        if k in seed_keys:
            stats["leak"] += 1; continue                  # decontaminate
        h = hashlib.md5(k.encode()).hexdigest()
        if h in seen:
            stats["dup"] += 1; continue                   # dedupe
        seen.add(h); kept.append(p)
    stats["kept"] = len(kept)
    return kept, stats


def main() -> None:
    seed = [json.loads(l) for l in SEED.read_text().splitlines()] if SEED.exists() else []
    print(f"=== §19 synthetic data | teacher={TEACHER} | seed={len(seed)} pairs ===")

    raw: list[dict] = []
    if _ollama_up():
        for t in TOPICS:
            p = teacher_pair(t)
            if p:
                raw.append(p)
                print(f"  + Q: {p['prompt'][:70]}")
    else:
        print("  [skip gen] Ollama not reachable on :11434 — demoing QC on a fixed sample.")
        raw = [{"prompt": "What does the Hartree-Fock method approximate?",   # a deliberate leak
                "completion": "It approximates the wavefunction as a Slater determinant."},
               {"prompt": "What is the variational principle?",
                "completion": "Any trial wavefunction gives an energy >= the true ground-state energy."},
               {"prompt": "What is the variational principle?",               # a deliberate dup
                "completion": "The energy of any trial wavefunction is an upper bound to the true energy."},
               {"prompt": "What is spin?", "completion": "too short"}]        # too short

    kept, stats = qc(raw, seed)
    print(f"\n  QC: in={stats['in']} -> kept={stats['kept']} "
          f"(dropped: {stats['leak']} leak, {stats['dup']} dup, {stats['short']} short)")
    OUT.write_text("\n".join(json.dumps(p) for p in kept))
    print(f"  saved -> {OUT.relative_to(ROOT.parent)}")
    print("  Reminder: a teacher's errors/style are inherited — judge a sample before training on it.")


if __name__ == "__main__":
    main()
