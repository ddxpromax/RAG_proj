# Operations

## Important Paths

```text
/root/RAG_proj/sustech-rag
/root/autodl-fs/sustech-rag
```

## Rebuild Core Data

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_pipeline_core.sh
python scripts/fetch_pdf_sources.py
python scripts/parse.py
python scripts/chunk.py
python scripts/build_bm25.py
python scripts/embed.py
python scripts/generate_eval.py
python scripts/evaluate.py --split dev
python scripts/evaluate.py --split test
python scripts/evaluate_generation.py --split test --mode hybrid_rerank --use-llm
python scripts/report_summary.py
```

`fetch_pdf_sources.py` discovers and downloads the configured public PDF sources,
then appends them to the raw manifest before parsing. Re-run parse, chunk, BM25,
and dense embedding after refreshing PDFs.

## Start Services

Terminal 1:

```bash
bash scripts/run_llm.sh
```

Terminal 2:

```bash
bash scripts/run_api.sh
```

Terminal 3:

```bash
bash scripts/run_ui.sh
```

The Gradio UI sends requests to the RAG API at `http://127.0.0.1:8080` by
default. Override with `RAG_API_BASE` only if the API runs elsewhere.

Download real local LLM weights:

```bash
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /root/autodl-fs/sustech-rag/models/generator \
  --max-workers 4

# Alternative helper:
python scripts/download_models.py --target generator --provider modelscope
bash scripts/run_llm.sh
```

When `model.safetensors` is present under `/root/autodl-fs/sustech-rag/models/generator`,
`run_llm.sh` uses the Transformers backend. Otherwise it starts the local extractive backend,
which is still OpenAI-compatible and suitable for API/UI smoke tests.

Download or refresh real local embedding weights:

```bash
python scripts/download_models.py --target embedding --provider modelscope
python scripts/embed.py
```

The dense index uses Qwen3-Embedding-0.6B from
`/root/autodl-fs/sustech-rag/models/embedding`. Re-run `scripts/embed.py` after
changing embedding weights, chunking, or source data.

Download or refresh real local reranker weights:

```bash
python scripts/download_models.py --target reranker --provider modelscope
```

The reranker uses Qwen3-Reranker-0.6B from
`/root/autodl-fs/sustech-rag/models/reranker`. The implementation scores query
and document pairs with Qwen's yes/no causal-LM formulation, then blends that
score with the original hybrid rank prior for stability on this dataset.

## Health Checks

```bash
curl http://127.0.0.1:8080/health
curl -I http://127.0.0.1:7860
python scripts/health_check.py
```

`health_check.py` reports the local generation backend as `transformers` or `extractive`.

## Known Environment Notes

- Docker is not available in the current AutoDL image, so local embedded Qdrant is used.
- `NO_PROXY` is set for localhost to avoid AutoDL proxy interference.
- Large model downloads from ModelScope/Hugging Face may be slow. The project remains usable with the local extractive backend until weights are downloaded.
- Embedded Qdrant keeps a local file lock; run direct Python retrieval/evaluation jobs sequentially unless the API service is mediating retrieval.
