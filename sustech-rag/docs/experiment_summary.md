# Experiment Summary

## Data Snapshot

- Documents: 285
- Chunks: 2649
- Source categories:

| category | documents | chunks |
| --- | --- | --- |
| global | 30 | 169 |
| library | 36 | 135 |
| student_affairs | 60 | 293 |
| undergraduate_teaching | 120 | 1889 |
| university_overview | 39 | 163 |

- Source file types:

| source_type | documents |
| --- | --- |
| html | 243 |
| pdf | 42 |

## Dev Retrieval Scores

| mode | doc_hit_at_5 | doc_hit_at_10 | chunk_hit_at_5 | chunk_hit_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- |
| dense | 1 | 1 | 0.800 | 0.800 | 0.878 |
| bm25 | 0.800 | 0.933 | 0.667 | 0.733 | 0.730 |
| hybrid | 1 | 1 | 0.867 | 0.933 | 0.917 |
| hybrid_rerank | 0.933 | 1 | 0.867 | 0.933 | 0.944 |

## Test Retrieval Scores

| mode | doc_hit_at_5 | doc_hit_at_10 | chunk_hit_at_5 | chunk_hit_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- |
| dense | 0.983 | 0.983 | 0.800 | 0.817 | 0.873 |
| bm25 | 0.933 | 0.967 | 0.883 | 0.883 | 0.871 |
| hybrid | 0.967 | 0.967 | 0.817 | 0.817 | 0.877 |
| hybrid_rerank | 0.917 | 0.967 | 0.817 | 0.817 | 0.876 |

## Current Notes

- Dense retrieval uses local Qwen3-Embedding-0.6B vectors stored in embedded Qdrant.
- Hybrid reranking uses local Qwen3-Reranker-0.6B causal scoring blended with the original hybrid rank prior.
- The generator service is OpenAI-compatible and local. It reports `backend=transformers` when Qwen2.5-0.5B-Instruct weights are present under `/root/autodl-fs/sustech-rag/models/generator`; otherwise it falls back to `backend=extractive` for smoke tests.
- The latest generation/refusal evaluation was run with the local Qwen2.5-0.5B-Instruct Transformers backend.

## Generation / Refusal Summary

{
  "split": "test",
  "mode": "hybrid_rerank",
  "cases": 65,
  "citation_correct_rate": 0.9166666666666666,
  "false_refusal_rate": 0.0,
  "refusal_accuracy": 1.0,
  "unanswerable_refusal_rate": 1.0
}
