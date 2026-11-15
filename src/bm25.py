"""
Phase 2 — BM25 from scratch (Okapi BM25, Robertson et al.)

For a query q with terms t, the score of document d is

    score(d, q) = sum over t in q of   IDF(t) * tf(t,d) * (k1 + 1)
                                     -----------------------------------------
                                     tf(t,d) + k1 * (1 - b + b * |d| / avgdl)

  tf(t,d)  how many times term t appears in d
  |d|      length of d in tokens; avgdl = average length across the corpus
  IDF(t)   = ln( (N - df + 0.5) / (df + 0.5) + 1 )     ("+1" keeps it positive)
           rare terms are worth more: "fiduciary" >> "act"
  k1 (~1.2-2.0)  TF saturation: the 10th "penalty" adds far less than the 1st
  b  (~0.75)     length normalisation: long sections don't win just by being long

Run:  py -3.8 src/bm25.py "time limit to reply to an RTI request"
"""
import math
import sys
from collections import Counter

from common import load_chunks, to_citations, tokenize


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.ids = [c["chunk_id"] for c in chunks]
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]   # term freqs per doc
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        N = len(chunks)
        df = Counter(t for tf in self.tfs for t in tf)                  # docs containing t
        self.idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        # inverted index: term -> list of doc indices. Only score docs sharing a term.
        self.index = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.index.setdefault(t, []).append(i)

    def score_doc(self, i, q_terms):
        tf, norm = self.tfs[i], self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query, k=10):
        q = tokenize(query)
        cand = {i for t in q for i in self.index.get(t, [])}
        scored = sorted(((self.ids[i], self.score_doc(i, q)) for i in cand),
                        key=lambda x: -x[1])
        return scored[:k]

    def explain(self, query, chunk_id):
        """Per-term contribution — use this to SEE why a doc ranked where it did."""
        i, q = self.ids.index(chunk_id), tokenize(query)
        return {t: round(self.score_doc(i, [t]), 3) for t in q}


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "time limit to reply to an RTI request"
    chunks = load_chunks()
    bm = BM25(chunks)
    print(f"query tokens: {tokenize(query)}\n")
    hits = bm.search(query, k=20)
    for cid, s in hits[:5]:
        print(f"{s:6.2f}  {cid:18s}  {bm.explain(query, cid)}")
    print("\nas citations:", [c for c, _ in to_citations(hits, k=5)])
