# RAG Project

Course project repository for a SUSTech campus knowledge-base RAG system.

The implementation lives in [`sustech-rag/`](sustech-rag/). It includes:

- official-source crawling and PDF ingestion
- HTML/PDF parsing, chunking, and indexing
- BM25, dense, hybrid, and reranked retrieval
- local Qwen-based embedding, reranking, and generation
- citation-aware answering and refusal handling
- FastAPI backend, Gradio demo, and evaluation scripts
- report, demo script, and presentation notes

## Quick Links

- [Project README](sustech-rag/README.md)
- [Final report](sustech-rag/docs/project_report.md)
- [Experiment summary](sustech-rag/docs/experiment_summary.md)
- [Demo script](sustech-rag/docs/demo_script.md)
- [Presentation notes](sustech-rag/docs/presentation_notes.md)
- [Operations guide](sustech-rag/docs/operations.md)

## Repository Layout

```text
.
├── references/          # project brief, handoff PDF, config screenshot
└── sustech-rag/         # code, configs, docs, tests, frontend
```

Large runtime artifacts are not committed. In the AutoDL setup they live under:

```text
/root/autodl-fs/sustech-rag
```

They include crawled raw data, normalized documents, chunks, BM25/Qdrant indexes,
evaluation outputs, and downloaded model weights.
