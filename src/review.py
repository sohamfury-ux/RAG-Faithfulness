"""
Phase 5 — Blind human verification of the eval set via POOLING.

Why pooling?  A gold label list written by one person has two failure modes:
  (a) a WRONG gold section              -> everyone is scored against a mistake
  (b) a MISSED valid section            -> a retriever that finds it is punished
Pooling attacks both: we gather candidate sections from every retriever + the
author's gold, shuffle them, hide the source, and let a second person judge each
candidate independently. This is how TREC-style test collections are built.

Usage (run in YOUR terminal - it is interactive):
  py -3.8 src/review.py pool            # build data/pool.json (loads the models once)
  py -3.8 src/review.py judge [N]       # judge N items (default 10), blind
  py -3.8 src/review.py report          # agreement between your judgments and the draft gold

"Relevant" = the section's own text contains information needed to answer the
question. A section that is merely on the same topic is NOT relevant.
"""
import json
import random
import sys
from collections import defaultdict

from common import ROOT, citation, load_chunks, to_citations
from validate_evalset import load_eval

POOL = ROOT / "data" / "pool.json"
JUDG = ROOT / "data" / "judgments.jsonl"
DEPTH = 4          # top-N citations taken from each retriever


def section_text():
    t = defaultdict(list)
    for c in load_chunks():
        t[citation(c)].append(c["text"])
    return {k: " ".join(v) for k, v in t.items()}


def build_pool():
    from hybrid import build_all
    R = build_all()
    pool = {}
    for it in load_eval():
        cands = set(it["gold"]) | set(it["acceptable"])
        for r in R.values():
            cands |= {c for c, _ in to_citations(r.search(it["question"], 40), DEPTH)}
        pool[it["id"]] = sorted(cands)
    POOL.write_text(json.dumps(pool, indent=1), encoding="utf-8")
    sizes = [len(v) for v in pool.values()]
    print(f"pool built: {len(pool)} questions, avg {sum(sizes)/len(sizes):.1f} candidates each")


def snippet(text, n=420):
    body = text.split("]", 1)[1].strip() if text.startswith("[") else text
    return body[:n].replace("\n", " ")


def judged_ids():
    if not JUDG.exists():
        return set()
    return {json.loads(l)["id"] for l in open(JUDG, encoding="utf-8") if l.strip()}


def judge(n):
    pool, text = json.loads(POOL.read_text(encoding="utf-8")), section_text()
    items = [i for i in load_eval() if i["id"] not in judged_ids()]
    random.Random(7).shuffle(items)
    print("Judge each candidate: does ITS OWN TEXT contain info needed to answer?")
    print("Enter numbers separated by spaces. Enter = none are relevant, s = skip, q = quit.\n")
    for k, it in enumerate(items[:n], 1):
        cands = list(pool[it["id"]])
        random.Random(sum(map(ord, it["id"]))).shuffle(cands)      # hide gold position
        print("=" * 78 + f"\n[{k}/{min(n, len(items))}] QUESTION: {it['question']}\n")
        for j, c in enumerate(cands, 1):
            print(f" {j:2d}) {c}\n      {snippet(text[c])}\n")
        ans = input("relevant> ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        try:
            chosen = [cands[int(x) - 1] for x in ans.split()]
        except (ValueError, IndexError):
            print("  (couldn't parse that, skipping)")
            continue
        with open(JUDG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": it["id"], "relevant": chosen}) + "\n")


def report():
    ev = {i["id"]: i for i in load_eval()}
    rows = [json.loads(l) for l in open(JUDG, encoding="utf-8") if l.strip()]
    agree = 0
    print(f"{len(rows)} items judged by you\n")
    for r in rows:
        it, you = ev[r["id"]], set(r["relevant"])
        gold, ok = set(it["gold"]), set(it["acceptable"])
        exact = you == gold
        agree += exact
        missing = gold - you                 # I said gold, you did not
        extra = you - gold - ok              # you said relevant, I never listed it
        if exact:
            continue
        print(f"{r['id']} [{it['type']}] {it['question']}")
        print(f"   my gold: {sorted(gold) or '(none: unanswerable)'}   yours: {sorted(you) or '(none)'}")
        if missing:
            print(f"   ⚠ you did NOT confirm my gold {sorted(missing)}  -> maybe my label is wrong")
        if extra:
            print(f"   ⚠ you found relevant {sorted(extra)} that I missed -> maybe a missing label")
        print()
    print(f"exact agreement: {agree}/{len(rows)} = {agree/max(len(rows),1):.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pool"
    if cmd == "pool":
        build_pool()
    elif cmd == "judge":
        judge(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "report":
        report()
