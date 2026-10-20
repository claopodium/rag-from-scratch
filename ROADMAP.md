# Roadmap

Current state: a working ingest/chat pipeline with dense-only retrieval.

The next phase is about **measurable retrieval quality** — building an evaluation harness first,
then improving retrieval against it.

## Phase 0 — Evaluation harness

- [ ] `Evaluator` class: load a JSONL question set, run retrieval, compute metrics
- [ ] Retrieval metrics: `Recall@k`, `MRR`, `nDCG@k`
- [ ] Ground truth stored as **character spans in the source text**, not chunk IDs, so labels
      survive changes to chunking parameters
- [ ] A reproducible `make eval` (or equivalent) that prints a results table

Planned datasets:

| Role | Dataset | Why |
|---|---|---|
| Primary | [MultiHop-RAG](https://arxiv.org/abs/2401.15391) | Multi-hop questions over a small news corpus (609 articles, 2,556 queries); retrieval, not generation, is the reported bottleneck (GPT-4: 0.56 with retrieved chunks vs 0.89 with gold evidence) |
| Chinese cross-check | RGB | Chinese RAG benchmark with four capability categories, including negative rejection |
| Retrieval sanity check | BEIR subsets (SciFact, NFCorpus) | Pure retrieval metrics, no generation model involved |

All three fit the current exhaustive-search implementation (≤ 10⁴–10⁵ chunks). Larger corpora —
HotpotQA full-wiki, full BEIR — require an ANN index first.

## Phase 1 — GPU build

- [ ] CUDA build documented and verified (Blackwell `sm_120`, CUDA ≥ 12.8)
- [ ] Larger generation model (7B–14B class)

## Phase 2 — Hybrid retrieval

- [ ] Inverted index with BM25
- [ ] Tokenization for CJK text (character bigrams, or reuse the model tokenizer)
- [ ] Score fusion with dense retrieval

## Phase 3 — Reranking

- [ ] Cross-encoder reranker as a second retrieval stage

## Phase 4 — Graph expansion

- [ ] Document-level entity/relation extraction (cheaper than chunk-level by an order of magnitude)
- [ ] Adjacency-list knowledge graph with BFS neighborhood expansion
- [ ] Graph-expanded retrieval, evaluated separately on the multi-hop subset

## Phase 5 — Scale (only if the corpus demands it)

- [ ] HNSW (or comparable) approximate nearest-neighbor index
- [ ] Benchmark recall vs. query latency
