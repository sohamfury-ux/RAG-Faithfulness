"""
Phase 3 — Dense retrieval with a bi-encoder (BAAI/bge-small-en-v1.5).

Idea: a neural net maps any text to a 384-dim vector so that texts with similar
MEANING land close together, even with zero shared words ("leak" ~ "failure to
protect data"). Retrieval = nearest neighbours by cosine similarity.

  bi-encoder: query and chunk are encoded SEPARATELY  -> chunk vectors can be
              precomputed once; a search is just one matrix multiply.
              (Phase 4's cross-encoder reads query+chunk TOGETHER: more accurate,
              but must run once per candidate — too slow for the whole corpus.)

Details that matter:
  * bge wants an instruction prefix on QUERIES only (asymmetric retrieval:
    short question vs long passage). Forgetting it is a classic silent bug.
  * normalize_embeddings=True -> vectors have length 1, so dot product = cosine.
  * max_seq_length 512 tokens; our chunks are <=250 words (~350 tokens), so
    nothing is truncated. Chunk size and the encoder's window must agree!

Run:  py -3.8 src/dense.py "can I get compensation if a company leaks my data"
"""
import hashlib
import sys

import numpy as np

from common import ROOT, load_chunks, to_citations

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / "data"


def _cache_path(chunks):
    # key the cache on the chunk TEXT, not just the count: same count + edited text
    # would otherwise silently serve stale vectors (a classic invisible bug).
    h = hashlib.sha1("\x1f".join(c["text"] for c in chunks).encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"emb_{MODEL.split('/')[-1]}_{h}.npy"


class Dense:
    def __init__(self, chunks, model_name=MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cuda")
        self.ids = [c["chunk_id"] for c in chunks]
        CACHE = _cache_path(chunks)
        if CACHE.exists():
            self.emb = np.load(CACHE)
        else:  # encode the corpus once, cache to disk
            self.emb = self.model.encode([c["text"] for c in chunks], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True)
            np.save(CACHE, self.emb)

    def search(self, query, k=10):
        q = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        sims = self.emb @ q                       # cosine similarity to every chunk
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


if __name__ == "__main__":
    from bm25 import BM25
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "time limit to reply to an RTI request",
        "punishment for identity theft",
        "can I get compensation if a company leaks my data",
        "what happens if my child's data is processed without parental consent",
        "can police read my WhatsApp messages",
        "I bought a defective phone, where do I complain",
    ]
    chunks = load_chunks()
    bm, de = BM25(chunks), Dense(chunks)
    for q in queries:
        b = [c for c, _ in to_citations(bm.search(q, 30), 5)]
        d = to_citations(de.search(q, 30), 5)
        print(f"\n### {q}")
        print(f"{'rank':4s}  {'BM25':16s}  DENSE (cosine)")
        for r in range(5):
            print(f"{r+1:<4d}  {b[r] if r < len(b) else '-':16s}  {d[r][0]:16s} {d[r][1]:.3f}")
