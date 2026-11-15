"""
Phase 5 — Validate the eval set against the corpus. Run after EVERY edit.

Checks that catch real labeling bugs:
  1. every gold/acceptable citation exists in the corpus (typos, cut sections)
  2. every item's verbatim `evidence` quote really appears in its gold section(s)
     (an evidence quote that is NOT found = I labeled from memory, not from source)
  3. unanswerable items have no gold; answerable items have >= 1
  4. duplicate questions / ids

Reports lexical overlap (question vs gold text) by question type. Why? If questions
share many words with their gold section, BM25 gets an unfair advantage: that is
LABELER BIAS, and it is the most common way an eval set silently favors one retriever.

Run:  py -3.8 src/validate_evalset.py
"""
import json
import re
import sys
from collections import Counter, defaultdict

from common import ROOT, citation, load_chunks, tokenize

EVAL = ROOT / "data" / "evalset.jsonl"


def norm(s):
    # punctuation -> space, so PDF artifacts ("twenty -one", "sub -section") still match
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s.lower())).strip()


MIN_EVIDENCE_WORDS = 4   # "electronic" or "mediation" would match almost anything: a fake check


def load_eval():
    return [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]


def main():
    chunks = load_chunks()
    text = defaultdict(str)
    for c in chunks:
        text[citation(c)] += " " + c["text"]
    items, errors = load_eval(), []

    ids = Counter(i["id"] for i in items)
    qs = Counter(i["question"].lower() for i in items)
    errors += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    errors += [f"duplicate question: {k}" for k, v in qs.items() if v > 1]

    overlap = defaultdict(list)
    for it in items:
        tag = f"{it['id']} ({it['type']})"
        for c in it["gold"] + it["acceptable"]:
            if c not in text:
                errors.append(f"{tag}: citation '{c}' not in corpus")
        if it["type"] == "unanswerable" and it["gold"]:
            errors.append(f"{tag}: unanswerable item has gold")
        if it["type"] != "unanswerable" and not it["gold"]:
            errors.append(f"{tag}: answerable item has no gold")
        if it["gold"] and all(c in text for c in it["gold"]):
            joined = norm(" ".join(text[c] for c in it["gold"]))
            for key in ("evidence", "evidence2"):
                if it.get(key) and norm(it[key]) not in joined:
                    errors.append(f"{tag}: {key} quote NOT FOUND in gold text: '{it[key]}'")
                if it.get(key) and len(norm(it[key]).split()) < MIN_EVIDENCE_WORDS:
                    errors.append(f"{tag}: {key} too short to be a real check: '{it[key]}'")
            if it["type"] == "multi" and len(it["gold"]) > 1 and not it.get("evidence2"):
                errors.append(f"{tag}: multi-section item needs evidence2 for its 2nd gold section")
            q, g = set(tokenize(it["question"])), set(tokenize(joined))
            overlap[it["type"]].append(len(q & g) / max(len(q), 1))

    print(f"{len(items)} items | " + " ".join(f"{t}={n}" for t, n in
          Counter(i['type'] for i in items).items()))
    print(f"split: {dict(Counter(i.get('split', '?') for i in items))}   "
          f"seen-by-author: {sum(i['seen'] for i in items)}   "
          f"status: {dict(Counter(i['status'] for i in items))}")
    acts = Counter(g.split()[0] for i in items for g in i["gold"][:1])
    print(f"first-gold act: {dict(acts)}   cross-act: "
          f"{sum(len({g.split()[0] for g in i['gold']}) > 1 for i in items)}")
    errors += [f"{i['id']}: missing split" for i in items if "split" not in i]
    print("\nlexical overlap, question tokens found in gold text (higher = easier for BM25):")
    for t, v in sorted(overlap.items()):
        print(f"  {t:12s} mean={sum(v)/len(v):.2f}  n={len(v)}")
    print(f"\n{len(errors)} problems")
    for e in errors:
        print("  ✗", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
