# SUSTech Campus RAG

This repository implements Project 1: SUSTech Campus Knowledge Base Construction with RAG.

## Storage Layout

From the repository root, code lives in:

```text
sustech-rag/
```

In the AutoDL environment used for this project, large runtime artifacts live in:

```text
/root/autodl-fs/sustech-rag
```

This keeps models, raw crawls, indexes, Qdrant storage, and logs off the 30GB system disk.

## First Commands

```bash
cd sustech-rag
python -m pip install -e ".[full,dev]"
make env-check
make init-dirs
```

Small crawl and BM25-only sanity path:

```bash
python scripts/crawl.py --source sustech_main
python scripts/parse.py
python scripts/chunk.py
python scripts/build_bm25.py
uvicorn sustech_rag.api.app:app --host 0.0.0.0 --port 8080
```

Convenience scripts:

```bash
bash scripts/run_pipeline_small.sh
bash scripts/run_pipeline_core.sh
python scripts/fetch_pdf_sources.py
bash scripts/run_api.sh
bash scripts/run_ui.sh
```

Full corpus rebuild with the current public SUSTech HTML and PDF sources:

```bash
bash scripts/run_pipeline_core.sh
python scripts/fetch_pdf_sources.py
python scripts/parse.py
python scripts/chunk.py
python scripts/build_bm25.py
python scripts/embed.py
python scripts/generate_eval.py
```

Recommended local runtime:

```bash
# Terminal 1: OpenAI-compatible local generation endpoint.
bash scripts/run_llm.sh

# Terminal 2: RAG API.
bash scripts/run_api.sh

# Terminal 3: Gradio demo.
bash scripts/run_ui.sh
```

The Gradio demo calls the RAG API on `http://127.0.0.1:8080` by default, so
embedded Qdrant is accessed through a single API process during demos.

If the AutoDL environment exports `HTTP_PROXY` or `HTTPS_PROXY`, keep localhost
traffic direct:

```bash
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

Evaluation:

```bash
python scripts/generate_eval.py
python scripts/evaluate.py --split dev
python scripts/evaluate.py --split test
python scripts/evaluate_generation.py --split test --mode hybrid_rerank --use-llm
python scripts/run_ablations.py --split test
```

For the full system, build dense embeddings, start the local generation service, then run the API and Gradio UI. Docker is not required in the current AutoDL environment because Qdrant uses local embedded storage.

Local LLM:

```bash
# Preferred on AutoDL/China network.
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /root/autodl-fs/sustech-rag/models/generator \
  --max-workers 4

# Python helper alternative.
python scripts/download_models.py --target generator --provider modelscope

bash scripts/run_llm.sh
```

If complete Transformers weights are present in `configs/models.yaml` `generator.local_path`,
`run_llm.sh` loads the real local model. If weights are absent or incomplete, it starts a local
OpenAI-compatible extractive fallback so the RAG API and UI remain fully runnable.

Dense local Qdrant path:

```bash
# Default uses local Qwen3-Embedding-0.6B vectors.
python scripts/download_models.py --target embedding --provider modelscope
python scripts/embed.py
```

The current `configs/models.yaml` points to
`/root/autodl-fs/sustech-rag/models/embedding`. The old hashing backend remains
available only as an offline fallback if `embedding.backend` is changed back to
`hashing`.

Local reranker:

```bash
python scripts/download_models.py --target reranker --provider modelscope
```

The current reranker uses `Qwen/Qwen3-Reranker-0.6B` with causal yes/no scoring,
blended conservatively with the original hybrid rank prior.

## Planned Modes

- `no_rag`: local LLM baseline without retrieval
- `dense`: Qdrant dense retrieval
- `bm25`: sparse lexical retrieval
- `hybrid`: Dense + BM25 + RRF
- `hybrid_rerank`: hybrid retrieval plus local Qwen3 reranker blending

The current implementation includes BM25, local Qdrant dense retrieval, hybrid RRF fusion, reranking, evidence sufficiency checks, cited generation, refusal handling, FastAPI, Gradio, and evaluation scripts.

The current data snapshot contains 285 normalized official documents: 243 HTML pages and 42 PDFs from the SUSTech undergraduate training-plan mirror, producing 2649 chunks.

## Delivery Docs

- `docs/README.md`
- `docs/project_report.md`
- `docs/presentation_notes.md`
- `docs/experiment_summary.md`
- `docs/ablation_report.md`
- `docs/presentation_tables.md`
- `docs/demo_script.md`
- `docs/project_report_outline.md`
- `docs/operations.md`
- `docs/final_checklist.md`
