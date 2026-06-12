# vLLM Startup

Recommended first pass for the 4090D instance:

```bash
export HF_HOME=/root/autodl-fs/sustech-rag/models/hf-cache
export MODELSCOPE_CACHE=/root/autodl-fs/sustech-rag/models/modelscope-cache

python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-fs/sustech-rag/models/generator \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.82 \
  --served-model-name sustech-rag-llm
```

If memory is tight, lower `--max-model-len` before changing the model.

## Transformers Fallback

If vLLM is not installed on the AutoDL image, use the local Transformers OpenAI-compatible server:

```bash
python scripts/download_models.py --target generator
bash scripts/run_llm.sh
```

It serves `/v1/chat/completions` on port `8000`, so the RAG backend can use the same `api_base`.

If model weights are not present under `models/generator`, the same script starts a local extractive
OpenAI-compatible generator. This keeps the RAG service fully local and testable while large model
weights are still downloading.
