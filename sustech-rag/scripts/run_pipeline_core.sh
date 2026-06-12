#!/usr/bin/env bash
set -euo pipefail

cd /root/RAG_proj/sustech-rag
python scripts/crawl.py \
  --source sustech_main \
  --source admissions_undergrad \
  --source tao_undergrad \
  --source osa_student_affairs \
  --source library \
  --source graduate_school \
  --source global
python scripts/parse.py
python scripts/fetch_api_sources.py
python scripts/chunk.py
python scripts/build_bm25.py
