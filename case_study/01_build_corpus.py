r"""
01_build_corpus.py — build the CPT corpus from Wikipedia (computational/quantum chemistry).

WHY THIS SCRIPT EXISTS, AND WHAT IT TEACHES
-------------------------------------------
This is the first running example and it makes research question #2 (SFT data availability)
concrete: raw domain text is *abundant*. A few dozen Wikipedia pages give hundreds of KB of clean
text in seconds, with no labelling. That is the fuel for **continued pretraining (CPT)**.

Because CPT is plain next-token prediction over raw text, the training signal is the **full causal
LM loss on every token** (see RESEARCH_NOTES §7). That is *why this corpus is a stream of raw text*
(a `{"text": ...}` dataset) and NOT prompt/completion pairs — there is no prompt to mask. Contrast
with SFT (02/05/06), where the data is prompt→completion and loss is computed on the **completion
only** (`completion_only_loss=True`). Keep that distinction in mind: the shape of the data here is
dictated by the loss we intend to compute downstream.

THE RECIPE (every setting is deliberate and documented — reproducibility is a hard requirement)
-----------------------------------------------------------------------------------------------
Source        : English Wikipedia via the MediaWiki API (action=query&prop=extracts&explaintext).
                Licence CC BY-SA — attributed in manifest.json and the book chapter.
User-Agent    : a descriptive UA string. Wikipedia returns HTTP 403 to bare/absent UAs; this is the
                single most common reason a naive scraper "works on my machine" but fails for others.
Politeness    : 0.3s sleep between network fetches.
Cleaning      : (1) MATH handling (see below); (2) Unicode NFKC normalization; (3) drop boilerplate
                tail sections (References / See also / External links / ...); (4) collapse whitespace.
Min length    : pages with < 200 chars after cleaning are treated as empty (disambiguation / wrong
                title / redirect stub) and SKIPPED with a warning so you can fix the title.
Deduplication : essential for CPT. Duplicated text wastes compute and *over-weights* whatever topic
                is repeated, biasing the model and inflating reported corpus size (D4, arXiv:2308.12284).
                Wikipedia redirects silently cause this (e.g. "Generalized gradient approximation" ->
                "Density functional theory"). We dedup on (a) resolved title and (b) exact content
                hash, keeping the first occurrence and reporting the rest.
Caching       : one plaintext file per *unique* page under data/corpus/ (skip-if-exists; --force refetch).
Manifest      : manifest.json records UA, source/licence, timestamp, every setting above, and per-page
                stats (resolved title, chars, words, #equations kept, content hash). Self-documenting.

MATH / EQUATION HANDLING (the crucial, domain-specific part)
------------------------------------------------------------
Quantum-chemistry pages are equation-dense, and `explaintext` renders each equation TWICE:
  1. a broken glyph-by-glyph Unicode dump (every symbol on its own short line: "Ψ\n(\nx\n,\nt\n)"), then
  2. the clean LaTeX source inside a `{\displaystyle ... }` blob.
Feeding (1) into CPT trains the model on noise. So we:
  - extract the LaTeX from every `{\displaystyle ...}` / `{\textstyle ...}` blob (balanced-brace scan,
    because LaTeX contains nested braces), normalize its internal whitespace, and re-insert it inline
    as `$ ... $` — a clean, standard representation;
  - delete the glyph-soup duplicate (runs of >=3 consecutive lines that each hold <=3 non-space chars;
    normal prose never lays words out one-per-line, so this does not eat real text).

TOKEN DECISION (documented so it is auditable): we add **NO new vocabulary tokens** for the math.
LaTeX is ASCII (`\ { } ^ _` + letters) and the model's existing BPE tokenizer already represents it.
Adding special tokens you don't need is discouraged (HF chat-templating docs: extra special tokens are
"often incorrect or duplicated, hurting performance") and would force a `resize_token_embeddings` step,
hurting reproducibility. Instead, CPT lets the *existing* embeddings adapt to the heavier LaTeX/math
distribution — which is exactly why the CPT LoRA target_modules include `embed_tokens` and `lm_head`
(see config.CPT). New *control* tokens (e.g. <THINKING>) would be a different story: those need
`tokenizer.add_special_tokens` + `model.resize_token_embeddings`, or Unsloth's `add_new_tokens(...)`
called BEFORE `get_peft_model()`.

Run: python case_study/01_build_corpus.py            (add --force to refetch from the network)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

UA = "llm-kickstart-case-study/0.1 (educational; contact: deeptij2007@gmail.com)"
API = "https://en.wikipedia.org/w/api.php"
MIN_CHARS = 200

# Tail sections that add noise to a CPT corpus — drop them and everything after.
_TAIL = re.compile(
    r"\n=+\s*(References|See also|External links|Further reading|Notes|Bibliography|Citations)\s*=+",
    re.IGNORECASE,
)
# Secondary sweep: a run of >=4 short/blank lines (<=3 non-space chars each, blanks allowed)
# catches any residual glyph soup not adjacent to a {\displaystyle} blob (e.g. trailing soup).
_GLYPH_SOUP = re.compile(r"(?m)(?:^[^\S\n]*\S{0,3}[^\S\n]*\n){4,}")


def _strip_trailing_glyph_run(seg: str) -> str:
    """Remove the trailing glyph-soup run from a segment that sits right before an equation.

    Wikipedia renders inline math as one symbol per line (blank lines interspersed) immediately
    before the {\\displaystyle} blob. We walk back over trailing lines that are blank or <=3
    non-space chars and drop them, stopping at the first real prose line (>3 chars). This is
    surgical: it only ever touches the soup adjacent to an equation, never prose elsewhere.
    """
    lines = seg.split("\n")
    j = len(lines)
    while j > 0 and len(lines[j - 1].strip()) <= 3:
        j -= 1
    tail_removed = len(lines) - j
    return "\n".join(lines[:j]) + ("\n" if tail_removed else "")


# ── Math extraction ───────────────────────────────────────────────────────────
def _extract_latex_blobs(text: str) -> tuple[str, int]:
    r"""Replace every {\displaystyle ...}/{\textstyle ...} blob with inline ` $ latex $ `, and
    strip the glyph-soup duplicate that precedes it.

    Uses a balanced-brace scan because LaTeX bodies contain nested {...}; a non-greedy regex
    to the first '}' would truncate the equation. Returns (new_text, num_equations).
    """
    out: list[str] = []
    i = 0
    n_eq = 0
    markers = ("{\\displaystyle", "{\\textstyle")
    while True:
        # find the next blob start
        nxt = min((p for p in (text.find(m, i) for m in markers) if p != -1), default=-1)
        if nxt == -1:
            out.append(text[i:])
            break
        out.append(_strip_trailing_glyph_run(text[i:nxt]))   # drop the rendered-glyph duplicate
        # scan to the matching close brace
        depth = 0
        k = nxt
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[nxt:k + 1]
        latex = body.split(None, 1)[1] if len(body.split(None, 1)) > 1 else ""
        latex = latex.rstrip("}").strip()
        latex = re.sub(r"\s+", " ", latex)
        out.append(f" $ {latex} $ " if latex else " ")
        n_eq += 1
        i = k + 1
    return "".join(out), n_eq


def clean(raw: str) -> tuple[str, int]:
    """Full cleaning pipeline. Returns (clean_text, num_equations_kept)."""
    text, n_eq = _extract_latex_blobs(raw)     # 1. keep LaTeX inline, strip adjacent glyph soup
    text = _GLYPH_SOUP.sub("", text)            # 2. secondary sweep for any residual soup
    text = unicodedata.normalize("NFKC", text)  # 3. Unicode normalization
    text = _TAIL.split(text, maxsplit=1)[0]     # 4. drop boilerplate tail sections
    text = re.sub(r"[ \t]+", " ", text)          # 5. collapse whitespace
    text = re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), n_eq


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_plaintext(title: str) -> tuple[str, str]:
    params = dict(action="query", format="json", prop="extracts",
                  explaintext=1, redirects=1, titles=title)
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")


def build_corpus(force: bool = False) -> dict:
    """Build (or load from cache) the deduped corpus. Returns the manifest dict.

    Callable from both the CLI (`main()`) and the notebook, so the logic lives in exactly one
    place — script and notebook can never drift out of sync.
    """
    config.set_all_seeds()
    manifest = {
        "user_agent": UA, "source": "en.wikipedia.org", "licence": "CC BY-SA 4.0",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"min_chars": MIN_CHARS, "math": "displaystyle->$latex$ + glyph-soup removal",
                     "normalization": "NFKC", "dedup": "resolved-title + content-hash",
                     "tail_sections_dropped": True, "added_tokens": "none (LaTeX uses existing BPE vocab)"},
        "env": config.record_env(), "pages": [],
    }
    by_hash: dict[str, str] = {}      # content hash -> first file that had it
    by_title: dict[str, str] = {}     # resolved title -> first query
    totals = dict(chars=0, words=0, equations=0)
    empties, dupes = [], []

    for title in config.WIKI_PAGES:
        path = config.CORPUS_DIR / f"{slugify(title)}.txt"
        if path.exists() and not force:
            txt = path.read_text()
            # each equation is " $ latex $ " -> two " $ " delimiters, so halve to get the count
            resolved, status, n_eq = title, "cached", txt.count(" $ ") // 2
        else:
            try:
                resolved, rawtext = fetch_plaintext(title)
                txt, n_eq = clean(rawtext)
            except Exception as e:
                print(f"  ✗ {title!r}: {type(e).__name__} {e}")
                empties.append(title)
                continue
            if len(txt) < MIN_CHARS:
                print(f"  ⚠ {title!r} -> {resolved!r}: {len(txt)} chars < {MIN_CHARS}, SKIPPED (fix title)")
                empties.append(title)
                time.sleep(0.3)
                continue
            status = "fetched"
            time.sleep(0.3)

        h = hashlib.sha256(txt.encode()).hexdigest()
        if h in by_hash:
            print(f"  ⊘ {title!r} -> {resolved!r}: duplicate content of {by_hash[h]!r} (redirect?), SKIPPED")
            dupes.append((title, by_hash[h]))
            if status == "cached":
                pass  # leave stale cache file; just don't include in corpus
            continue
        if resolved in by_title:
            print(f"  ⊘ {title!r} resolves to {resolved!r}, already taken by {by_title[resolved]!r}, SKIPPED")
            dupes.append((title, by_title[resolved]))
            continue

        if status == "fetched":
            path.write_text(txt)
        by_hash[h], by_title[resolved] = title, title
        words = len(txt.split())
        totals["chars"] += len(txt); totals["words"] += words; totals["equations"] += n_eq
        manifest["pages"].append(dict(query=title, resolved=resolved, file=path.name,
                                      chars=len(txt), words=words, equations=n_eq,
                                      sha256=h[:16], status=status))
        print(f"  ✓ {title!r:42} {len(txt):>7} chars / {words:>6} words / {n_eq:>3} eqs [{status}]")

    (config.CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = len(manifest["pages"])
    print(f"\nCorpus: {ok} unique pages | {totals['chars']:,} chars | ~{totals['words']:,} words "
          f"| ~{totals['words'] * 4 // 3:,} tokens est. | {totals['equations']:,} equations kept")
    print(f"Saved to {config.CORPUS_DIR}  |  manifest.json written (records every setting above)")
    if dupes:
        print(f"\n{len(dupes)} duplicate(s) skipped (redirects collapsing onto an existing page):")
        for q, kept in dupes:
            print(f"  - {q!r} == {kept!r}")
    if empties:
        print(f"\n{len(empties)} page(s) need a corrected title in config.WIKI_PAGES: {empties}")
    print("\nThis is the abundant side of the data story. Next: 02_data_availability.py")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the CPT corpus from Wikipedia.")
    ap.add_argument("--force", action="store_true", help="refetch from the network even if cached")
    build_corpus(force=ap.parse_args().force)


if __name__ == "__main__":
    main()
