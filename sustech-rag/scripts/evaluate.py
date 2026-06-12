from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import load_paths
from sustech_rag.retrieval.service import RetrievalService


def load_cases(split: str) -> list[dict]:
    paths = load_paths()
    path = Path(paths["data"]["eval"]) / f"{split}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reciprocal_rank(hits, relevant_chunks: set[str], relevant_docs: set[str]) -> float:
    for index, hit in enumerate(hits, start=1):
        if hit.chunk_id in relevant_chunks or hit.doc_id in relevant_docs:
            return 1.0 / index
    return 0.0


def hit_at(hits, relevant_chunks: set[str], relevant_docs: set[str], k: int, level: str) -> int:
    for hit in hits[:k]:
        if level == "chunk" and hit.chunk_id in relevant_chunks:
            return 1
        if level == "doc" and hit.doc_id in relevant_docs:
            return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "demo"], default="dev")
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()

    cases = load_cases(args.split)
    service = RetrievalService()
    paths = load_paths()
    out_dir = Path(paths["data"]["eval"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{args.split}_retrieval_outputs.jsonl"
    summary_rows = []

    with raw_path.open("w", encoding="utf-8") as raw:
        for mode in args.modes:
            totals = {
                "answerable": 0,
                "doc_hit_at_5": 0,
                "doc_hit_at_10": 0,
                "chunk_hit_at_5": 0,
                "chunk_hit_at_10": 0,
                "mrr_at_10": 0.0,
            }
            for case in cases:
                if not case.get("answerable", True):
                    continue
                relevant_chunks = set(case.get("relevant_chunk_ids") or [])
                relevant_docs = set(case.get("relevant_doc_ids") or [])
                result = service.retrieve(case["question"], mode=mode, context_top_k=10)
                hits = result["hits"]
                totals["answerable"] += 1
                totals["doc_hit_at_5"] += hit_at(hits, relevant_chunks, relevant_docs, 5, "doc")
                totals["doc_hit_at_10"] += hit_at(hits, relevant_chunks, relevant_docs, 10, "doc")
                totals["chunk_hit_at_5"] += hit_at(hits, relevant_chunks, relevant_docs, 5, "chunk")
                totals["chunk_hit_at_10"] += hit_at(hits, relevant_chunks, relevant_docs, 10, "chunk")
                totals["mrr_at_10"] += reciprocal_rank(hits[:10], relevant_chunks, relevant_docs)
                raw.write(
                    json.dumps(
                        {
                            "mode": mode,
                            "question_id": case["question_id"],
                            "question": case["question"],
                            "hits": [hit.model_dump(mode="json") for hit in hits],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            n = max(1, totals["answerable"])
            summary_rows.append(
                {
                    "split": args.split,
                    "mode": mode,
                    "answerable": totals["answerable"],
                    "doc_hit_at_5": totals["doc_hit_at_5"] / n,
                    "doc_hit_at_10": totals["doc_hit_at_10"] / n,
                    "chunk_hit_at_5": totals["chunk_hit_at_5"] / n,
                    "chunk_hit_at_10": totals["chunk_hit_at_10"] / n,
                    "mrr_at_10": totals["mrr_at_10"] / n,
                }
            )

    csv_path = out_dir / f"{args.split}_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(row)
    print("raw", raw_path)
    print("scores", csv_path)


if __name__ == "__main__":
    main()
