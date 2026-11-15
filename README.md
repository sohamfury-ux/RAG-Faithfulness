# RAG with measured faithfulness — Indian statutes

A retrieval-augmented question-answering system over four Indian Acts, built to answer one
question honestly: **when the system cites a section, does that section actually support the
claim?** It compares four retrievers (BM25, dense, hybrid, reranker), then scores generated
answers claim by claim, not by similarity to a reference answer.

| Acts (corpus) | Source |
|---|---|
| Right to Information Act, 2005 | https://cic.gov.in/sites/default/files/RTI-Act_English.pdf |
| Information Technology Act, 2000 | https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf |
| Consumer Protection Act, 2019 | https://www.indiacode.nic.in/indiacode/bitstream/123456789/15256/1/eng201935.pdf |
| Digital Personal Data Protection Act, 2023 | https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf |

Copies are kept in `data/raw/` because government links move (one of the first links tried had
already died). These are official public documents; check the source sites for current versions.

## Pipeline

```
PDF ─ingest─▶ section-aware chunks ─▶ BM25 ┐
                                  ─▶ dense ├─▶ hybrid (RRF) ─▶ cross-encoder reranker
                                           ┘
questions + gold sections (eval set) ─▶ retrieval metrics (Recall@k, MRR, nDCG, paired CIs)
top-5 sections ─▶ small local LLM ─▶ answer with [S1]-style citations ─▶ parsed into claims
each claim ─▶ NLI judge vs (a) the section it cites, (b) any retrieved section ─▶ faithfulness
```

The citation unit is **(Act, section)**, so retrieval, labels and faithfulness are all scored
at the level a reader would check.

| Step | File | Idea |
|---|---|---|
| Ingest | `src/ingest.py` | Section-aware chunking; schedules are their own citation units |
| BM25 | `src/bm25.py` | Hand-written Okapi BM25 with an inverted index |
| Dense | `src/dense.py` | `BAAI/bge-small-en-v1.5`, embeddings cached by content hash |
| Hybrid / rerank | `src/hybrid.py` | Reciprocal Rank Fusion; `BAAI/bge-reranker-base` over top 30 |
| Eval set | `data/evalset.jsonl`, `src/evalset_test_v1.py`, `src/validate_evalset.py` | Gold sections with verbatim evidence quotes, checked by script |
| Human review | `src/review.py` | Blind pooled judging to catch wrong/missing labels |
| Retrieval eval | `src/eval_retrieval.py`, `src/paired.py` | Metrics + paired bootstrap confidence intervals |
| Generation | `src/generate.py` | Grounded answers (`Qwen2.5-1.5B-Instruct`) with citation parsing |
| Faithfulness | `src/faithfulness.py`, `src/validate_judge.py` | NLI judge, validated against hand labels |

## The eval set (150 questions)

- **`dev` (43):** written first; the author saw retriever output on these. Debugging only.
- **`test` (107):** written from section text *before* any retriever ran on them
  (`src/evalset_test_v1.py`). This is the headline split. Do not edit; add new ids instead.
- Types: direct, paraphrase, multi-section (incl. cross-Act), unanswerable (e.g. "as may be
  prescribed" traps), and one stale-law case (IT Act s.66A, struck down in 2015).
- `py -3.8 src/validate_evalset.py` checks every citation exists and every evidence quote
  (>= 4 words) is really in its gold section. It caught mislabeled items during development.

## Results so far (read the caveats)

**Retrieval, held-out test split (95 answerable questions):**

| | Recall@5 | MRR |
|---|---|---|
| BM25 | 0.84 | 0.71 |
| Dense | 0.87 | 0.73 |
| Hybrid | 0.88 | 0.77 |
| Rerank | 0.83 | 0.74 |

- Only one paired difference is statistically clear on the full test set: hybrid > BM25 on MRR.
- The ranking **depends on how questions are worded.** On true paraphrases (low word overlap
  with the statute, n=21) dense beats BM25 clearly (+0.24 MRR); on statute-worded questions
  BM25 and hybrid match or beat dense. Report results by overlap bucket, not one average.
- All four retrievers miss list-shaped sections (e.g. DPDP s.15) and none ever abstains.

**Faithfulness (43 dev questions, older corpus, validated large NLI judge):** about 38–51% of
claims are supported by some retrieved section, only 12–22% carry a citation, and the four
retrievers cannot be separated on this metric. The small generator, not retrieval, is the
bottleneck.

### Known limitations
- **Labels are single-annotator drafts.** Run `py -3.8 src/review.py pool` then `judge`, and
  merge disagreements, before trusting any number here.
- **Faithfulness numbers are stale.** They were generated before the schedule fix and before the
  107 test questions existed. Regenerate with `generate.py retrieve`, `generate.py generate all`,
  then `faithfulness.py`.
- The NLI judge was validated on only 30 hand-labeled claims (accuracy about 0.80).
- Faithful is not the same as true: a claim matching an outdated section counts as supported.

## Reproduce

Python 3.8 with CUDA (developed on an RTX 2050, 4 GB). Versions used are in `requirements.txt`.

```
py -3.8 src/ingest.py                    # PDFs -> data/chunks.jsonl
py -3.8 src/validate_evalset.py          # check the eval set
py -3.8 src/eval_retrieval.py            # retrieval metrics (test split is the headline)
py -3.8 src/generate.py retrieve         # step 1: save contexts (retrievers on GPU)
py -3.8 src/generate.py generate all     # step 2: write answers (LLM on GPU)
py -3.8 src/faithfulness.py              # step 3: score every claim
```

On Windows set `PYTHONIOENCODING=utf-8`. Dense embeddings are cached in `data/` and rebuilt
automatically when the chunk text changes.

## Repository layout
- `src/` code · `data/raw/` source PDFs · `data/chunks.jsonl` corpus · `data/evalset.jsonl` eval set
- `data/answers_*.jsonl`, `data/faithfulness_details.jsonl`, `data/results_retrieval.json`: saved outputs
- `data/judge_labels.json`: hand labels used to validate the NLI judge
