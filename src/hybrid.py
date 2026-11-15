"""
Phase 4 — Hybrid retrieval (RRF) + cross-encoder reranking.

1) Reciprocal Rank Fusion (Cormack et al. 2009)
       RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))        k = 60
   Uses only RANKS, so BM25's 0..22 scores and cosine's 0.5..0.7 never need to
   be put on the same scale. A doc ranked well by BOTH retrievers wins; a doc
   ranked #1 by one and absent from the other still gets a decent score.
   k dampens the head: with k=60, rank 1 vs rank 2 is 1/61 vs 1/62 — tiny. RRF
   rewards agreement more than any single retriever's confidence.

2) Cross-encoder reranker
   The bi-encoder (Phase 3) squeezed each chunk into one vector BEFORE seeing the
   query. A cross-encoder reads "[query] [SEP] [chunk]" together, so every query
   token attends to every chunk token ("police" <-> "any agency of the
   appropriate Government"). Much more accurate — but one forward pass PER
   (query, chunk) pair, so we only run it on the top-N candidates.

   This is the classic two-stage design:
       cheap, high-RECALL first stage (hybrid, top 30)
    -> expensive, high-PRECISION second stage (reranker, top 5)
   The reranker can only reorder what stage 1 found: recall@30 of stage 1 is a
   hard ceiling on the whole pipeline.

Run:  py -3.8 src/hybrid.py ["your query"]
"""
import sys

from common import load_chunks, to_citations

RERANKER = "BAAI/bge-reranker-base"


def rrf(rankings, k=60, top=None):
    """rankings: list of ranked lists [(chunk_id, score), ...] -> fused ranked list."""
    fused = {}
    for ranked in rankings:
        for rank, (cid, _) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return out[:top] if top else out


class Hybrid:
    def __init__(self, bm25, dense, depth=50):
        self.bm25, self.dense, self.depth = bm25, dense, depth

    def search(self, query, k=10):
        return rrf([self.bm25.search(query, self.depth),
                    self.dense.search(query, self.depth)], top=k)


class Reranked:
    def __init__(self, first_stage, chunks, n_candidates=30, model_name=RERANKER):
        from sentence_transformers import CrossEncoder
        self.first, self.n = first_stage, n_candidates
        self.text = {c["chunk_id"]: c["text"] for c in chunks}
        self.ce = CrossEncoder(model_name, device="cuda", max_length=512,
                               automodel_args={"torch_dtype": "auto"})
        self.ce.model.half()   # fp16: halves memory on the 4GB GPU, same rankings

    def search(self, query, k=10):
        cands = [cid for cid, _ in self.first.search(query, self.n)]
        scores = self.ce.predict([(query, self.text[c]) for c in cands], batch_size=8)
        return sorted(zip(cands, map(float, scores)), key=lambda x: -x[1])[:k]


def build_all(chunks=None):
    """Construct all four retrievers once. Reused by the evaluation phases."""
    from bm25 import BM25
    from dense import Dense
    chunks = chunks or load_chunks()
    bm, de = BM25(chunks), Dense(chunks)
    hy = Hybrid(bm, de)
    return {"bm25": bm, "dense": de, "hybrid": hy, "rerank": Reranked(hy, chunks)}


if __name__ == "__main__":
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "time limit to reply to an RTI request",
        "punishment for identity theft",
        "can I get compensation if a company leaks my data",
        "what happens if my child's data is processed without parental consent",
        "can police read my WhatsApp messages",
        "I bought a defective phone, where do I complain",
    ]
    R = build_all()
    for q in queries:
        cols = {name: [c for c, _ in to_citations(r.search(q, 40), 5)] for name, r in R.items()}
        print(f"\n### {q}")
        print("rank  " + "".join(f"{n:16s}" for n in cols))
        for i in range(5):
            print(f"{i+1:<4d}  " + "".join(f"{(v[i] if i < len(v) else '-'):16s}" for v in cols.values()))
