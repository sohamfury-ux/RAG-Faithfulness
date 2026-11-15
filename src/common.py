"""Shared helpers: loading chunks, tokenizing, collapsing chunk hits to citations."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"


def load_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# Small stopword list. Note what is NOT here: "not", "no", "shall", "may".
# In law those words carry meaning ("shall" = mandatory, "may" = discretionary).
STOP = set("""a an the of to in and or for on by with as at be is are was were been this that
these those it its from which such any under any other than into upon""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def stem(w):
    """Deliberately crude suffix stripping (no NLTK needed).
    'penalties' -> 'penalti', 'penalty' -> 'penalti'  : now they match.
    Over-stemming risk: 'information' and 'inform' collide — acceptable here."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s", "y"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("i" if suf in ("ies", "ied", "y") else "")
    return w


def tokenize(text):
    # "66A" -> "66a" stays one token, so section numbers are searchable.
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP]


def citation(chunk_or_id):
    """'IT2000-s69#1' -> 'IT2000 s.69'. The unit we evaluate and cite at."""
    cid = chunk_or_id["chunk_id"] if isinstance(chunk_or_id, dict) else chunk_or_id
    doc, rest = cid.split("-s", 1)
    return f"{doc} s.{rest.split('#')[0]}"


def to_citations(ranked, k=None):
    """Collapse ranked chunk hits [(chunk_id, score)] into ranked unique citations.
    A section's score = its best chunk's score (max-pooling)."""
    seen, out = set(), []
    for cid, score in ranked:
        c = citation(cid)
        if c not in seen:
            seen.add(c)
            out.append((c, score))
            if k and len(out) == k:
                break
    return out
