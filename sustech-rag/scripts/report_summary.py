from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import PROJECT_ROOT, load_paths


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_Not generated yet._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            try:
                numeric = float(value)
                value = str(int(numeric)) if numeric.is_integer() else f"{numeric:.3f}"
            except Exception:
                pass
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    paths = load_paths()
    data = Path(paths["data"]["normalized"]) / "documents.jsonl"
    chunks_path = Path(paths["data"]["chunks"]) / "chunks.jsonl"
    docs = [json.loads(line) for line in data.open("r", encoding="utf-8") if line.strip()]
    chunks = [json.loads(line) for line in chunks_path.open("r", encoding="utf-8") if line.strip()]
    doc_categories = Counter(doc["category"] for doc in docs)
    doc_types = Counter(doc["source_type"] for doc in docs)
    chunk_categories = Counter(chunk["metadata"].get("category") for chunk in chunks)
    eval_dir = Path(paths["data"]["eval"]) / "results"
    dev_scores = read_csv(eval_dir / "dev_scores.csv")
    test_scores = read_csv(eval_dir / "test_scores.csv")
    generation_summary_path = eval_dir / "test_generation_hybrid_rerank_summary.json"
    generation_summary = json.loads(generation_summary_path.read_text(encoding="utf-8")) if generation_summary_path.exists() else {}

    out = PROJECT_ROOT / "docs" / "experiment_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# Experiment Summary",
                "",
                "## Data Snapshot",
                "",
                f"- Documents: {len(docs)}",
                f"- Chunks: {len(chunks)}",
                "- Source categories:",
                "",
                table(
                    [{"category": k, "documents": doc_categories[k], "chunks": chunk_categories[k]} for k in sorted(doc_categories)],
                    ["category", "documents", "chunks"],
                ),
                "- Source file types:",
                "",
                table(
                    [{"source_type": k, "documents": doc_types[k]} for k in sorted(doc_types)],
                    ["source_type", "documents"],
                ),
                "## Dev Retrieval Scores",
                "",
                table(dev_scores, ["mode", "doc_hit_at_5", "doc_hit_at_10", "chunk_hit_at_5", "chunk_hit_at_10", "mrr_at_10"]),
                "## Test Retrieval Scores",
                "",
                table(test_scores, ["mode", "doc_hit_at_5", "doc_hit_at_10", "chunk_hit_at_5", "chunk_hit_at_10", "mrr_at_10"]),
                "## Current Notes",
                "",
                "- Dense retrieval uses local Qwen3-Embedding-0.6B vectors stored in embedded Qdrant.",
                "- Hybrid reranking uses local Qwen3-Reranker-0.6B causal scoring blended with the original hybrid rank prior.",
                "- The generator service is OpenAI-compatible and local. It reports `backend=transformers` when Qwen2.5-0.5B-Instruct weights are present under `/root/autodl-fs/sustech-rag/models/generator`; otherwise it falls back to `backend=extractive` for smoke tests.",
                "- The latest generation/refusal evaluation was run with the local Qwen2.5-0.5B-Instruct Transformers backend.",
                "",
                "## Generation / Refusal Summary",
                "",
                json.dumps(generation_summary, ensure_ascii=False, indent=2) if generation_summary else "_Not generated yet._",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
