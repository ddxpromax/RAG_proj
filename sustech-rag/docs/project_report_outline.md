# Project Report Outline

## 1. Project Background and Objectives

This project builds a SUSTech campus knowledge-base Q&A system with Retrieval-Augmented Generation. The system addresses scattered campus information across official websites, teaching offices, admissions pages, student affairs, library pages, graduate school pages, and international collaboration pages.

## 2. Dataset / Data Source Introduction

Sources include:

- SUSTech main website
- Teaching Affairs Office
- Student Affairs Office
- Library
- Undergraduate admissions API
- Graduate School API
- Global engagement website

Artifacts:

- Raw archive and manifest under `/root/autodl-fs/sustech-rag/data`
- Normalized documents: `data/normalized/documents.jsonl`
- Chunks: `data/chunks/chunks.jsonl`

See `docs/experiment_summary.md` for the current data scale.

## 3. Model and Method Design

Retrieval stack:

- Dense local Qdrant index
- BM25 lexical index with jieba tokenization
- Reciprocal Rank Fusion
- Local Qwen3-Reranker-0.6B with hybrid rank-prior blending
- Evidence sufficiency and refusal check

Generation:

- OpenAI-compatible local LLM client
- Transformers local fallback server script
- Evidence-extractive fallback when model weights are unavailable

## 4. Experimental Process

Pipeline:

```text
source registry -> crawl/API fetch -> parse -> normalize -> chunk -> BM25/Dense index -> retrieve -> rerank -> answer
```

Main experiment modes:

- Dense
- BM25
- Hybrid
- Hybrid+Rerank

## 5. Result Display

Use the tables in `docs/experiment_summary.md`.

## 6. Problem Analysis

Observed issues:

- Dynamic websites need API-specific ingestion.
- Model weight downloads can be slow in the AutoDL environment.
- Hashing embedding remains available as a no-download fallback but the final system uses Qwen3-Embedding-0.6B.
- Evidence sufficiency checks are required for unanswerable questions.

## 7. Summary

The project delivers a reproducible campus RAG system with traceable data, multiple retrieval modes, source citations, refusal handling, and evaluation outputs.
