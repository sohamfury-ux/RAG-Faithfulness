"""Paired cluster bootstrap on faithfulness: resample QUESTIONS, compare retrievers on the same ones."""
import json, numpy as np
from collections import defaultdict
from common import ROOT
from faithfulness import metrics
rows = defaultdict(dict)
for l in open(ROOT/"data"/"faithfulness_details.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["answerable"]: rows[r["retriever"]][r["id"]] = r
ids = sorted(rows["rerank"])
rng = np.random.default_rng(0)
print(f"{'A - B':16s}{'metric':15s}{'diff':>7s}   95% CI")
for a, b in [("rerank","bm25"),("rerank","hybrid"),("rerank","dense"),("dense","bm25"),("hybrid","dense")]:
    for k in ("grounded_rate","cite_f1"):
        diffs = []
        for _ in range(1500):
            s = rng.choice(ids, len(ids))
            diffs.append(metrics([rows[a][i] for i in s])[k] - metrics([rows[b][i] for i in s])[k])
        d0 = metrics([rows[a][i] for i in ids])[k] - metrics([rows[b][i] for i in ids])[k]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        print(f"{a+' - '+b:16s}{k:15s}{d0:+7.3f}   [{lo:+.3f},{hi:+.3f}]{'  *' if lo>0 or hi<0 else ''}")
