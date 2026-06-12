PYTHON ?= python

.PHONY: env-check init-dirs crawl parse normalize chunk embed index serve-api serve-ui eval-dev eval-test test clean

env-check:
	$(PYTHON) scripts/env_check.py

init-dirs:
	$(PYTHON) scripts/init_dirs.py

download-embedding:
	$(PYTHON) scripts/download_models.py --target embedding

download-generator:
	$(PYTHON) scripts/download_models.py --target generator

crawl:
	$(PYTHON) scripts/crawl.py --config configs/sources.yaml

parse:
	$(PYTHON) scripts/parse.py

normalize:
	$(PYTHON) scripts/normalize.py

fetch-api:
	$(PYTHON) scripts/fetch_api_sources.py

chunk:
	$(PYTHON) scripts/chunk.py

embed:
	$(PYTHON) scripts/embed.py

index:
	$(PYTHON) scripts/build_bm25.py

eval-set:
	$(PYTHON) scripts/generate_eval.py

serve-api:
	uvicorn sustech_rag.api.app:app --host 0.0.0.0 --port $${RAG_API_PORT:-8080} --reload

serve-llm:
	bash scripts/run_llm.sh

serve-ui:
	$(PYTHON) frontend/app.py

eval-dev:
	$(PYTHON) scripts/evaluate.py --split dev

eval-test:
	$(PYTHON) scripts/evaluate.py --split test

eval-generation:
	$(PYTHON) scripts/evaluate_generation.py --split test

report-summary:
	$(PYTHON) scripts/report_summary.py

health:
	$(PYTHON) scripts/health_check.py

test:
	pytest -q

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
