# Ablation Report

This report records low-risk ablations run on the fixed corpus, chunks, BM25 index, and Qdrant dense index. It does not rebuild data or change the production demo configuration.

## Retrieval Mode Ablation

| experiment | doc_hit_at_5 | doc_hit_at_10 | chunk_hit_at_5 | chunk_hit_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- |
| retrieval_bm25 | 0.933 | 0.967 | 0.883 | 0.883 | 0.871 |
| retrieval_dense | 0.983 | 0.983 | 0.800 | 0.817 | 0.873 |
| retrieval_hybrid | 0.967 | 0.967 | 0.817 | 0.817 | 0.877 |
| retrieval_hybrid_rerank | 0.917 | 0.967 | 0.817 | 0.817 | 0.876 |

## Generation / Refusal Ablation

| experiment | use_llm | citation_correct_rate | false_refusal_rate | refusal_accuracy | unanswerable_refusal_rate |
| --- | --- | --- | --- | --- | --- |
| generation_hybrid_rerank_extractive | False | 0.917 | 0.000 | 1.000 | 1.000 |
| generation_hybrid_rerank_llm_cached | True | 0.917 | 0.000 | 1.000 | 1.000 |

## Notes

- Retrieval ablation compares sparse, dense, hybrid fusion, and hybrid plus reranking.
- Generation ablation compares evidence-extractive answering with the cached local-LLM generation evaluation when available.
- Evidence sufficiency remains enabled in all generation rows, because it is part of the safety-critical RAG answer path.
