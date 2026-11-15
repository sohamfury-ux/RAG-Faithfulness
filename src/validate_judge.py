"""
Phase 8b — Validate the NLI judge itself against human labels.

A faithfulness number is only as good as the judge producing it. Here we compare
judge variants to hand labels (1 = the source text supports the claim, 0 = not).

  py -3.8 src/validate_judge.py sample    # draw + print claims with their source text
  py -3.8 src/validate_judge.py eval      # score judge variants vs data/judge_labels.json
"""
import json
import random
import sys

import numpy as np

from common import ROOT
from faithfulness import Judge, clean_claim, windows
from validate_evalset import load_eval

SAMPLE = ROOT / "data" / "judge_sample.json"
LABELS = ROOT / "data" / "judge_labels.json"


def sample(n=30, seed=3):
    ev = {i["id"]: i for i in load_eval()}
    pool = []
    for name in ("dense", "rerank", "hybrid"):
        for l in open(ROOT / "data" / f"answers_{name}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            gold = set(ev[r["id"]]["gold"])
            src = next((s for s in r["context"] if s["citation"] in gold), None)
            if src and not r["abstained"]:
                for c in r["claims"]:
                    pool.append({"qid": r["id"], "retriever": name, "claim": c["sentence"],
                                 "citation": src["citation"], "premise": src["text"]})
    random.Random(seed).shuffle(pool)
    pool = pool[:n]
    for i, p in enumerate(pool):
        p["n"] = i + 1
    SAMPLE.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in pool:
        print(f"\n#{p['n']} [{p['qid']}/{p['retriever']}] SOURCE {p['citation']}:\n  {p['premise'][:1000]}")
        print(f"  >>> CLAIM: {clean_claim(p['claim'])}")


def evaluate():
    pool = json.loads(SAMPLE.read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in json.loads(LABELS.read_text()).items()}
    pool = [p for p in pool if p["n"] in labels]
    y = np.array([labels[p["n"]] for p in pool])
    print(f"{len(pool)} labeled claims: {int(y.sum())} supported, {int((1-y).sum())} not\n")
    configs = [("deberta-v3-base", "cross-encoder/nli-deberta-v3-base", False),
               ("deberta-v3-base+clean", "cross-encoder/nli-deberta-v3-base", True),
               ("deberta-v3-LARGE+clean", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", True)]
    for label, model, clean in configs:
        try:
            j = Judge(model, clean)
        except Exception as e:                                   # model unavailable
            print(f"{label}: could not load ({type(e).__name__})")
            continue
        s = np.array([j.entail(windows(p["premise"]), p["claim"]) for p in pool])
        print(f"== {label}")
        for tau in (0.5, 0.3, 0.1):
            pred = s >= tau
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            acc = (pred == y).mean()
            print(f"   tau={tau}: acc={acc:.2f}  precision={prec:.2f}  recall={rec:.2f}  (FP={fp} FN={fn})")
        del j
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    sample() if (sys.argv[1] if len(sys.argv) > 1 else "sample") == "sample" else evaluate()
