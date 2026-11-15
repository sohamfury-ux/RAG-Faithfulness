"""Paired bootstrap: is retriever A better than B on the SAME questions?
Resamples questions, compares mean(A_i - B_i). If the 95% interval excludes 0, the
difference is unlikely to be luck of which questions we happened to write."""
import json, itertools, numpy as np
from common import ROOT
R = json.loads((ROOT/"data"/"results_retrieval.json").read_text())["results"]
ids = list(R["bm25"])
rng = np.random.default_rng(0)
print(f"{'A - B':16s}{'metric':9s}{'mean diff':>10s}   95% CI           wins/losses/ties")
for a, b in [("dense","hybrid"),("rerank","hybrid"),("rerank","dense"),("dense","bm25"),("hybrid","bm25")]:
    for m in ("mrr","recall5"):
        d = np.array([R[a][i][m]-R[b][i][m] for i in ids])
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs,[2.5,97.5])
        sig = "  *" if lo>0 or hi<0 else ""
        print(f"{a+' - '+b:16s}{m:9s}{d.mean():10.3f}   [{lo:+.3f},{hi:+.3f}]  {int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}{sig}")
