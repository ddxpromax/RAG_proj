# Ablation Report

This report records low-risk ablations run on the fixed corpus, chunks, BM25 index, and Qdrant dense index. It does not rebuild data or change the production demo configuration.

## Retrieval Mode Ablation

| experiment | doc_hit_at_5 | doc_hit_at_10 | chunk_hit_at_5 | chunk_hit_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- |
| retrieval_bm25 | 0.933 | 0.967 | 0.883 | 0.883 | 0.871 |
| retrieval_dense | 0.983 | 0.983 | 0.800 | 0.817 | 0.873 |
| retrieval_hybrid | 0.967 | 0.967 | 0.817 | 0.817 | 0.877 |
| retrieval_hybrid_rerank | 0.917 | 0.967 | 0.817 | 0.817 | 0.876 |

## Reranker Blend-Weight Ablation

The Qwen reranker score is blended with the original hybrid rank prior. This ablation reuses the same traced reranker candidates and recalculates the final ranking for different weights, without changing the production API configuration.

| experiment | qwen_score_weight | doc_hit_at_5 | doc_hit_at_10 | chunk_hit_at_5 | chunk_hit_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| reranker_weight_0 | 0.000 | 0.967 | 0.967 | 0.817 | 0.817 | 0.877 |
| reranker_weight_0.25 | 0.250 | 0.917 | 0.967 | 0.817 | 0.817 | 0.876 |
| reranker_weight_0.5 | 0.500 | 0.917 | 0.967 | 0.833 | 0.833 | 0.875 |
| reranker_weight_0.75 | 0.750 | 0.917 | 0.967 | 0.833 | 0.833 | 0.874 |
| reranker_weight_1 | 1.000 | 0.900 | 0.983 | 0.833 | 0.833 | 0.825 |

## Generation / Refusal Ablation

| experiment | use_llm | citation_correct_rate | false_refusal_rate | refusal_accuracy | unanswerable_refusal_rate |
| --- | --- | --- | --- | --- | --- |
| generation_hybrid_rerank_extractive | False | 0.917 | 0.000 | 1.000 | 1.000 |
| generation_hybrid_rerank_llm_cached | True | 0.917 | 0.000 | 1.000 | 1.000 |

## Notes

- Retrieval ablation compares sparse, dense, hybrid fusion, and hybrid plus reranking.
- Reranker blend-weight ablation checks how strongly to trust the Qwen yes/no reranker score relative to the original hybrid rank prior.
- In this corpus, increasing Qwen reranker weight slightly improves chunk-level hit rate at 0.5 and 0.75, but it lowers MRR. A full Qwen-only score reaches the best doc_hit@10 but the worst MRR, so the production system keeps reranking conservative rather than over-trusting the small reranker.
- Generation ablation compares evidence-extractive answering with the cached local-LLM generation evaluation when available.
- Evidence sufficiency remains enabled in all generation rows, because it is part of the safety-critical RAG answer path.
