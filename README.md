# RAG with measured faithfulness — Indian statutes

Corpus: RTI Act 2005, IT Act 2000, Consumer Protection Act 2019, DPDP Act 2023 (data/raw).
Python: `py -3.8` (has torch+CUDA, sentence-transformers, pypdf already installed).

## Roadmap (learn-by-building: each phase ends with a quiz)
| Phase | Build | Key idea | Status |
|---|---|---|---|
| 1 | `src/ingest.py` — PDF → section-aware chunks with citation IDs | citation unit = (act, section) | done |
| 2 | `src/bm25.py` — BM25 from scratch | lexical retrieval, IDF, length norm | done |
| 3 | `src/dense.py` — bge-small embeddings | semantic retrieval, vocabulary mismatch | done |
| 4 | `src/hybrid.py` — Reciprocal Rank Fusion + cross-encoder reranker | fusion, bi- vs cross-encoder | done |
| 5 | `src/evalset/` — labeled eval set (draft → human verify) | **the hard part**: gold sections, hard negatives, unanswerables | 150 items: dev 43 + held-out test 107 (draft labels) |
| 6 | `src/eval_retrieval.py` — Recall@k, MRR, nDCG per retriever | measuring retrieval | re-run on 150 (test = headline) |
| 7 | `src/generate.py` — answers with inline citations `[RTI2005 s.7]` | grounded generation (free LLM) | done (172 answers saved) |
| 8 | `src/faithfulness.py` — citation precision/recall via NLI | faithfulness ≠ answer similarity | done (judge validated on 30 hand labels) |

Run: `py -3.8 src/ingest.py` → `data/chunks.jsonl`

## Eval set (data/evalset.jsonl)
- `dev` (43): original questions; the author has seen retriever output on them. Use for debugging only.
- `test` (107): written from section text before any retriever ran on them (`src/evalset_test_v1.py`). **Report this.** Don't edit; add new ids instead.
- Validate after every change: `py -3.8 src/validate_evalset.py` (citations exist, >=4-word evidence quotes found in gold text).
- Labels are still single-annotator drafts: run `py -3.8 src/review.py pool` then `judge`.
