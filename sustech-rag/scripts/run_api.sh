#!/usr/bin/env bash
set -euo pipefail

cd /root/RAG_proj/sustech-rag
export HF_HOME="${HF_HOME:-/root/autodl-fs/sustech-rag/models/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-fs/sustech-rag/models/modelscope-cache}"
export PYTHONPATH="${PYTHONPATH:-/root/RAG_proj/sustech-rag/src}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0}"

uvicorn sustech_rag.api.app:app --host 0.0.0.0 --port "${RAG_API_PORT:-8080}"
