#!/usr/bin/env bash
set -euo pipefail

cd /root/RAG_proj/sustech-rag
python scripts/crawl.py --source sustech_main
python scripts/parse.py
python scripts/chunk.py
python scripts/build_bm25.py

