"""
Phase 6 — Score the four retrievers against the eval set (citation level).

Retrievers return CHUNKS; we collapse them to unique (act, section) citations
(max-pooling) because the citation is what a user/judge sees and what Phase 8 scores.

Metrics (answerable questions only; unanswerable ones have no gold to retrieve):
  Hit@k     did ANY gold section appear in the top k?             (yes/no)
  Recall@k  what FRACTION of the gold sections are in the top k?   (matters for multi-section Qs)
  MRR       1 / rank of the first gold section                     (rewards being near the top)
  nDCG@5    graded: gold=2, acceptable=1, discounted by log2(rank+1), normalised by the ideal ordering
Because n is small (~39) we also bootstrap 95% confidence intervals: if two
retrievers' intervals overlap heavily, we cannot claim one is better.

Run:  py -3.8 src/eval_retrieval.py
"""
import json
import math
import sys

import numpy as np

from common import ROOT, to_citations
from validate_evalset import load_eval

OUT = ROOT / "data" / "results_retrieval.json"
NAMES = ["bm25", "dense", "hybrid", "rerank"]


def score_item(ranked, gold, acceptable):
    gold_s, rel = set(gold), {**{a: 1 for a in acceptable}, **{g: 2 for g in gold}}
    first = next((r for r, c in enumerate(ranked, 1) if c in gold_s), None)
    dcg = sum(rel.get(c, 0) / math.log2(r + 1) for r, c in enumerate(ranked[:5], 1))
    ideal = sorted(rel.values(), reverse=True)[:5]
    idcg = sum(v / math.log2(r + 1) for r, v in enumerate(ideal, 1))
    return {
        "hit1": float(first == 1),
        "hit5": float(first is not None and first <= 5),
        "recall5": len(gold_s & set(ranked[:5])) / len(gold_s),
        "recall10": len(gold_s & set(ranked[:10])) / len(gold_s),
        "mrr": 1.0 / first if first else 0.0,
        "ndcg5": dcg / idcg if idcg else 0.0,
        "first_rank": first,
    }


def boot_ci(vals, n=2000, seed=0):
    rng, v = np.random.default_rng(seed), np.asarray(vals)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def table(title, results, items, metrics=("hit1", "hit5", "recall5", "recall10", "mrr", "ndcg5")):
    print(f"\n{title}  (n={len(items)})")
    print(f"{'':8s}" + "".join(f"{m:>10s}" for m in metrics))
    for n in NAMES:
        row = [np.mean([results[n][i['id']][m] for i in items]) for m in metrics]
        print(f"{n:8s}" + "".join(f"{v:10.3f}" for v in row))


def main():
    from hybrid import build_all
    R = build_all()
    items = load_eval()
    answerable = [i for i in items if i["gold"]]
    results = {n: {} for n in NAMES}
    ranked_all = {n: {} for n in NAMES}
    stage1 = {}
    for it in items:
        for n in NAMES:
            ranked = [c for c, _ in to_citations(R[n].search(it["question"], 30))]
            ranked_all[n][it["id"]] = ranked
            if it["gold"]:
                results[n][it["id"]] = score_item(ranked, it["gold"], it["acceptable"])
        # the reranker's ceiling: how much gold is even in hybrid's top-30 chunks?
        if it["gold"]:
            cand = {c for c, _ in to_citations(R["hybrid"].search(it["question"], 30))}
            stage1[it["id"]] = len(set(it["gold"]) & cand) / len(it["gold"])

    judged = (ROOT / "data" / "judgments.jsonl").exists()
    print(f"labels: {'DRAFT + human judgments exist (not yet merged)' if judged else 'DRAFT ONLY (not human-verified)'}")

    test = [i for i in answerable if i.get("split") == "test"]
    dev = [i for i in answerable if i.get("split") == "dev"]
    table("HEADLINE: TEST split (held out, written before any retriever saw it)", results, test)
    print("\n  95% bootstrap CI (test)   MRR              Recall@5")
    for n in NAMES:
        m = boot_ci([results[n][i["id"]]["mrr"] for i in test])
        r = boot_ci([results[n][i["id"]]["recall5"] for i in test])
        print(f"    {n:8s}             [{m[0]:.2f}, {m[1]:.2f}]     [{r[0]:.2f}, {r[1]:.2f}]")
    print("\n  PAIRED differences on test (A - B; * = 95% CI excludes 0)")
    for a, b in [("dense", "bm25"), ("dense", "hybrid"), ("rerank", "dense"), ("rerank", "hybrid"), ("hybrid", "bm25")]:
        for m in ("mrr", "recall5"):
            d = np.array([results[a][i["id"]][m] - results[b][i["id"]][m] for i in test])
            lo, hi = boot_ci(d)
            print(f"    {a+' - '+b:16s}{m:8s}{d.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]"
                  f"{'  *' if lo > 0 or hi < 0 else ''}")
    table("DEV split (author has seen retriever output on these)", results, dev)

    # objective difficulty: how many question words appear in the gold text?
    from show_sections import section_texts
    from common import tokenize
    T = section_texts()
    def overlap(it):
        q = set(tokenize(it["question"]))
        return len(q & set(tokenize(" ".join(T[c] for c in it["gold"])))) / max(len(q), 1)
    for lo, hi, name in [(0, .35, "LOW overlap (<0.35): true paraphrase"),
                         (.35, .6, "MID overlap (0.35-0.6)"),
                         (.6, 1.01, "HIGH overlap (>=0.6): statute wording")]:
        sub = [i for i in test if lo <= overlap(i) < hi]
        if sub:
            table(f"test, {name}", results, sub, metrics=("hit5", "recall5", "mrr"))
    multi = [i for i in test if i["type"] == "multi"]
    table("test, MULTI-section", results, multi, metrics=("hit5", "recall5", "recall10", "mrr"))

    print(f"\nreranker ceiling: mean gold coverage inside hybrid's top-30 = "
          f"{np.mean(list(stage1.values())):.3f}  (rerank recall@10 cannot exceed this)")

    misses = [it for it in test if all((results[n][it["id"]]["first_rank"] or 99) > 5 for n in NAMES)]
    print(f"\nTEST questions where ALL FOUR miss the top 5 ({len(misses)}): first-gold rank (- = beyond 30)")
    for it in misses:
        ranks = [results[n][it["id"]]["first_rank"] for n in NAMES]
        print(f"  {it['id']} " + " ".join(f"{(r or '-'):>3}" for r in ranks)
              + f"  gold={it['gold']}  {it['question'][:70]}")

    print("\nUNANSWERABLE (test): #1 result per retriever (retrieval never abstains on its own)")
    for it in items:
        if not it["gold"] and it.get("split") == "test":
            print(f"  {it['id']} {it['question'][:48]:48s} " +
                  " | ".join(f"{n}:{ranked_all[n][it['id']][0]}" for n in NAMES))

    OUT.write_text(json.dumps({"results": results, "ranked": ranked_all}, indent=1), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
