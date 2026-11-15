"""
Phase 8 — Citation-level faithfulness.

Answer similarity asks "does the answer look like the reference?"  Faithfulness asks
"is every claim supported by the evidence the answer POINTS AT?"  Per claim we ask two
different questions with an NLI (entailment) model:

  supported_by_cited : does at least one of the sections it CITES entail the claim?
  supported_by_any   : does ANY retrieved source entail it?   (= grounded)

Reading the pair:
  grounded + cited-supports   -> good, verifiable
  grounded + cited-doesn't    -> RIGHT FACT, WRONG CITATION  (misattribution)
  grounded + no citation      -> true but unverifiable for the reader
  not grounded                -> UNSUPPORTED claim (hallucination, or outside knowledge)

Metrics per retriever (micro over claims):
  grounded_rate     share of claims supported by some retrieved source
  cite_rate         share of claims carrying any citation
  cite_precision    of cited claims, share whose citation really supports them
  cite_recall       of grounded claims, share that cite a section that supports them
  cite_f1           harmonic mean of the two
  gold_cite_acc     of cited sections, share that are in the eval set's gold/acceptable
NLI premise = 80-word sliding windows of each source (NLI models are trained on short
premises; long ones dilute the signal); we take the best window.

Run:  py -3.8 src/faithfulness.py
"""
import json
import sys

import numpy as np

from common import ROOT
from validate_evalset import load_eval

import re

NLI = "cross-encoder/nli-deberta-v3-base"
# base judge under-reported support (recall 0.38 vs hand labels); large judge: acc 0.80, P/R 0.77.
# See validate_judge.py. Default to the validated one.
LARGE = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NAMES = ["bm25", "dense", "hybrid", "rerank"]
TAU = 0.5           # entailment probability needed to count a claim as supported
WIN, STRIDE = 80, 40
DETAIL = ROOT / "data" / "faithfulness_details.jsonl"


_SRC = re.compile(r"\b(according to|as per|per|from|in|see)\s+(the\s+)?sources?\s*(\d+|S\d+)?\s*,?", re.I)
_BULLET = re.compile(r"\s*\n\s*[-*\u2022]?\s*")


def clean_claim(t):
    """Strip generation debris that confuses an NLI judge but isn't part of the claim:
    'According to Source ,' leftovers, markdown bullets/newlines, doubled spaces."""
    t = _SRC.sub("", t)
    t = _BULLET.sub(" ", t)
    t = re.sub(r"\bSources?\s*\d*\b", "", t)
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" ,;")


def windows(text):
    body = text.split("]", 1)[1].strip() if text.startswith("[") else text
    w = body.split()
    if len(w) <= WIN:
        return [" ".join(w)]
    return [" ".join(w[i:i + WIN]) for i in range(0, len(w) - STRIDE, STRIDE)]


class Judge:
    def __init__(self, model_name=None, clean=True):
        import torch
        global NLI
        NLI = model_name or NLI
        self.clean = clean
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI).half().cuda().eval()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(k for k, v in labels.items() if v == "entailment")
        self.cache = {}

    def entail(self, premises, claim):
        """max entailment probability of `claim` over a list of premise windows"""
        if self.clean:
            claim = clean_claim(claim)
        todo = [p for p in premises if (p, claim) not in self.cache]
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            enc = self.tok(batch, [claim] * len(batch), truncation=True, max_length=512,
                           padding=True, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1)[:, self.ent].cpu().tolist()
            for p, pr in zip(batch, probs):
                self.cache[(p, claim)] = pr
        return max(self.cache[(p, claim)] for p in premises)


def score_retriever(name, judge, ev):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"answers_{name}.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        item = ev[r["id"]]
        wins = {s["citation"]: windows(s["text"]) for s in r["context"]}
        gold_ok = set(item["gold"]) | set(item["acceptable"])
        in_ctx = bool(set(item["gold"]) & set(wins))
        claims = []
        for c in r["claims"]:
            sup = {cit: judge.entail(w, c["sentence"]) for cit, w in wins.items()}
            best_any = max(sup.values()) if sup else 0.0
            cited_scores = [sup[x] for x in c["cited"] if x in sup]
            claims.append({
                "sentence": c["sentence"], "cited": c["cited"], "support": sup,
                "grounded": best_any >= TAU,
                "cited_supports": bool(cited_scores) and max(cited_scores) >= TAU,
                "supporting_sources": [x for x, p in sup.items() if p >= TAU],
                "cited_in_gold": [x in gold_ok for x in c["cited"]],
            })
        out.append({"id": r["id"], "retriever": name, "abstained": r["abstained"],
                    "answerable": bool(item["gold"]), "gold_in_context": in_ctx,
                    "claims": claims, "answer": r["answer"], "question": r["question"]})
    return out


def metrics(rows):
    cl = [c for r in rows for c in r["claims"]]
    cited = [c for c in cl if c["cited"]]
    grounded = [c for c in cl if c["grounded"]]
    prec = np.mean([c["cited_supports"] for c in cited]) if cited else float("nan")
    rec = np.mean([c["cited_supports"] or bool(set(c["cited"]) & set(c["supporting_sources"]))
                   for c in grounded]) if grounded else float("nan")
    gc = [g for c in cited for g in c["cited_in_gold"]]
    return {
        "claims": len(cl),
        "grounded_rate": len(grounded) / len(cl) if cl else float("nan"),
        "cite_rate": len(cited) / len(cl) if cl else float("nan"),
        "cite_precision": prec, "cite_recall": rec,
        "cite_f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
        "gold_cite_acc": np.mean(gc) if gc else float("nan"),
    }


def boot(rows, keys, n=1000, seed=0):
    """cluster bootstrap: resample QUESTIONS (claims inside one answer are correlated)"""
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n):
        samp = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        m = metrics(samp)
        ms.append([m[k] for k in keys])
    ms = np.array(ms, dtype=float)
    return np.nanpercentile(ms, [2.5, 97.5], axis=0).T


def main():
    ev = {i["id"]: i for i in load_eval()}
    judge = Judge(sys.argv[1] if len(sys.argv) > 1 else LARGE)
    allrows = {n: score_retriever(n, judge, ev) for n in NAMES}
    with open(DETAIL, "w", encoding="utf-8") as f:
        for n in NAMES:
            for r in allrows[n]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    keys = ["grounded_rate", "cite_rate", "cite_precision", "cite_recall", "cite_f1", "gold_cite_acc"]
    print(f"NLI judge: {NLI}, tau={TAU}\n")
    print(f"{'':8s}{'claims':>7s}" + "".join(f"{k:>16s}" for k in keys))
    for n in NAMES:
        m = metrics([r for r in allrows[n] if r["answerable"]])
        print(f"{n:8s}{m['claims']:7d}" + "".join(f"{m[k]:16.3f}" for k in keys))
    print("\n(answerable questions only; 95% cluster-bootstrap CIs)")
    for n in NAMES:
        rows = [r for r in allrows[n] if r["answerable"]]
        ci = boot(rows, keys)
        print(f"  {n:8s}" + "  ".join(f"{k[:9]}[{lo:.2f},{hi:.2f}]" for k, (lo, hi) in zip(keys, ci)))

    print("\nDoes retrieval success carry through?  (answerable, non-abstained)")
    print(f"{'':8s}{'gold in ctx':>14s}{'grounded':>10s}{'cite_f1':>9s}   |{'gold NOT in ctx':>17s}{'grounded':>10s}")
    for n in NAMES:
        rows = [r for r in allrows[n] if r["answerable"] and r["claims"]]
        a, b = [r for r in rows if r["gold_in_context"]], [r for r in rows if not r["gold_in_context"]]
        ma, mb = metrics(a), metrics(b)
        print(f"{n:8s}{len(a):>14d}{ma['grounded_rate']:10.2f}{ma['cite_f1']:9.2f}   |{len(b):>17d}{mb['grounded_rate']:10.2f}")

    print("\nCLAIM DIAGNOSIS (share of all claims):")
    print(f"{'':8s}{'ok(cited+sup)':>15s}{'wrong-cite':>12s}{'uncited-true':>14s}{'UNSUPPORTED':>13s}")
    for n in NAMES:
        cl = [c for r in allrows[n] if r["answerable"] for c in r["claims"]]
        N = len(cl)
        ok = sum(c["cited_supports"] for c in cl)
        wrong = sum(bool(c["cited"]) and not c["cited_supports"] and c["grounded"] for c in cl)
        unc = sum(not c["cited"] and c["grounded"] for c in cl)
        uns = sum(not c["grounded"] for c in cl)
        print(f"{n:8s}{ok/N:15.2f}{wrong/N:12.2f}{unc/N:14.2f}{uns/N:13.2f}")

    print("\nABSTENTION")
    for n in NAMES:
        un = [r for r in allrows[n] if not r["answerable"]]
        an = [r for r in allrows[n] if r["answerable"]]
        print(f"  {n:8s} correct abstain on unanswerable {sum(r['abstained'] for r in un)}/{len(un)}"
              f"   false abstain on answerable {sum(r['abstained'] for r in an)}/{len(an)}")
        # unanswerable but answered: how many of those claims were still 'grounded'?
    print(f"\ndetails -> {DETAIL}")


if __name__ == "__main__":
    sys.exit(main())
