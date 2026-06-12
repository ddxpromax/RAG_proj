from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import load_paths
from sustech_rag.retrieval.service import RetrievalService


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'OK' if ok else 'FAIL'} {name} {detail}")
    return ok


def main() -> None:
    paths = load_paths()
    root = Path(paths["storage_root"])
    checks = []
    details: dict[str, Any] = {}
    docs = Path(paths["data"]["normalized"]) / "documents.jsonl"
    chunks = Path(paths["data"]["chunks"]) / "chunks.jsonl"
    bm25 = Path(paths["indexes"]["bm25"]) / "bm25.pkl"
    qdrant = Path(paths["indexes"]["qdrant_storage"]) / "collection" / "sustech_chunks_v1" / "storage.sqlite"
    test_scores = Path(paths["data"]["eval"]) / "results" / "test_scores.csv"
    gen_scores = Path(paths["data"]["eval"]) / "results" / "test_generation_hybrid_rerank_summary.json"
    checks.append(check("documents", docs.exists() and count_jsonl(docs) > 100, str(count_jsonl(docs)) if docs.exists() else "missing"))
    checks.append(check("chunks", chunks.exists() and count_jsonl(chunks) > 500, str(count_jsonl(chunks)) if chunks.exists() else "missing"))
    if bm25.exists():
        with bm25.open("rb") as f:
            payload = pickle.load(f)
        checks.append(check("bm25", len(payload.get("chunks", [])) == count_jsonl(chunks), str(len(payload.get("chunks", [])))))
    else:
        checks.append(check("bm25", False, "missing"))
    checks.append(check("qdrant_local", qdrant.exists(), str(qdrant)))
    checks.append(check("test_scores", test_scores.exists(), str(test_scores)))
    checks.append(check("generation_scores", gen_scores.exists(), str(gen_scores)))
    try:
        result = RetrievalService().retrieve("南方科技大学2026年本科招生章程有哪些要点？", mode="hybrid_rerank")
        checks.append(check("hybrid_rerank_query", bool(result["hits"]), f"hits={len(result['hits'])}"))
    except Exception as exc:
        if "already accessed by another instance of Qdrant client" in str(exc):
            try:
                response = httpx.post(
                    "http://127.0.0.1:8080/retrieve",
                    json={"question": "南方科技大学2026年本科招生章程有哪些要点？", "mode": "hybrid_rerank"},
                    timeout=10,
                    trust_env=False,
                )
                payload = response.json()
                checks.append(
                    check(
                        "hybrid_rerank_query",
                        response.status_code == 200 and bool(payload.get("hits")),
                        f"via_api hits={len(payload.get('hits', []))}",
                    )
                )
            except Exception as api_exc:
                checks.append(check("hybrid_rerank_query", False, f"{exc}; api fallback failed: {api_exc}"))
        else:
            checks.append(check("hybrid_rerank_query", False, str(exc)))
    try:
        response = httpx.get("http://127.0.0.1:8000/health", timeout=2, trust_env=False)
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        details["local_llm"] = payload
        backend = payload.get("backend", "unknown")
        checks.append(check("local_llm_endpoint", response.status_code == 200, f"backend={backend}"))
    except Exception as exc:
        checks.append(check("local_llm_endpoint", False, f"not running: {exc}"))
    summary = {"ok": all(checks), "storage_root": str(root), "details": details}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
