from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import PROJECT_ROOT, load_paths


RETRIEVAL_MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def load_cases(split: str) -> list[dict[str, Any]]:
    path = Path(load_paths()["data"]["eval"]) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reciprocal_rank(hits: list[dict], relevant_chunks: set[str], relevant_docs: set[str]) -> float:
    for index, hit in enumerate(hits, start=1):
        if hit.get("chunk_id") in relevant_chunks or hit.get("doc_id") in relevant_docs:
            return 1.0 / index
    return 0.0


def hit_at(hits: list[dict], relevant_chunks: set[str], relevant_docs: set[str], k: int, level: str) -> int:
    for hit in hits[:k]:
        if level == "chunk" and hit.get("chunk_id") in relevant_chunks:
            return 1
        if level == "doc" and hit.get("doc_id") in relevant_docs:
            return 1
    return 0


def score_generation_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = response.get("answer") or ""
    citations = response.get("citations") or []
    answerable = bool(case.get("answerable", True))
    relevant_docs = set(case.get("relevant_doc_ids") or [])
    relevant_chunks = set(case.get("relevant_chunk_ids") or [])
    citation_ids = {c.get("chunk_id") for c in citations} | {c.get("doc_id") for c in citations}
    citation_correct = bool((relevant_docs | relevant_chunks) & citation_ids) if answerable else True
    refused = response.get("evidence_status") == "insufficient_evidence" or "无法根据" in answer
    return {
        "question_id": case["question_id"],
        "answerable": answerable,
        "evidence_status": response.get("evidence_status"),
        "has_citation": bool(citations),
        "citation_correct": citation_correct,
        "refused": refused,
        "refusal_correct": (not answerable and refused) or (answerable and not refused),
        "answer_preview": answer[:160].replace("\n", " "),
    }


def evaluate_retrieval(
    client: httpx.Client,
    api_base: str,
    cases: list[dict[str, Any]],
    split: str,
    modes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    answerable_cases = [case for case in cases if case.get("answerable", True)]
    for mode in modes:
        totals = {
            "doc_hit_at_5": 0,
            "doc_hit_at_10": 0,
            "chunk_hit_at_5": 0,
            "chunk_hit_at_10": 0,
            "mrr_at_10": 0.0,
        }
        for case in answerable_cases:
            response = client.post(
                f"{api_base}/retrieve",
                json={"question": case["question"], "mode": mode, "context_top_k": 10},
            )
            response.raise_for_status()
            hits = response.json()["hits"]
            relevant_chunks = set(case.get("relevant_chunk_ids") or [])
            relevant_docs = set(case.get("relevant_doc_ids") or [])
            totals["doc_hit_at_5"] += hit_at(hits, relevant_chunks, relevant_docs, 5, "doc")
            totals["doc_hit_at_10"] += hit_at(hits, relevant_chunks, relevant_docs, 10, "doc")
            totals["chunk_hit_at_5"] += hit_at(hits, relevant_chunks, relevant_docs, 5, "chunk")
            totals["chunk_hit_at_10"] += hit_at(hits, relevant_chunks, relevant_docs, 10, "chunk")
            totals["mrr_at_10"] += reciprocal_rank(hits[:10], relevant_chunks, relevant_docs)
        n = max(1, len(answerable_cases))
        rows.append(
            {
                "split": split,
                "experiment": f"retrieval_{mode}",
                "mode": mode,
                "answerable": n,
                "doc_hit_at_5": totals["doc_hit_at_5"] / n,
                "doc_hit_at_10": totals["doc_hit_at_10"] / n,
                "chunk_hit_at_5": totals["chunk_hit_at_5"] / n,
                "chunk_hit_at_10": totals["chunk_hit_at_10"] / n,
                "mrr_at_10": totals["mrr_at_10"] / n,
            }
        )
    return rows


def evaluate_generation(
    client: httpx.Client,
    api_base: str,
    cases: list[dict[str, Any]],
    split: str,
    mode: str,
    use_llm: bool,
    experiment: str,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        response = client.post(
            f"{api_base}/chat",
            json={"question": case["question"], "mode": mode, "use_llm": use_llm},
        )
        response.raise_for_status()
        row = score_generation_case(case, response.json())
        row["question"] = case["question"]
        row["experiment"] = experiment
        rows.append(row)
    answerable_rows = [row for row in rows if row["answerable"]]
    unanswerable_rows = [row for row in rows if not row["answerable"]]
    return {
        "split": split,
        "experiment": experiment,
        "mode": mode,
        "use_llm": use_llm,
        "cases": len(rows),
        "citation_correct_rate": sum(row["citation_correct"] for row in answerable_rows) / max(1, len(answerable_rows)),
        "false_refusal_rate": sum(row["refused"] for row in answerable_rows) / max(1, len(answerable_rows)),
        "refusal_accuracy": sum(row["refusal_correct"] for row in rows) / max(1, len(rows)),
        "unanswerable_refusal_rate": sum(row["refused"] for row in unanswerable_rows) / max(1, len(unanswerable_rows)),
    }


def load_cached_llm_summary(split: str, mode: str) -> dict[str, Any] | None:
    path = Path(load_paths()["data"]["eval"]) / "results" / f"{split}_generation_{mode}_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "split": split,
        "experiment": f"generation_{mode}_llm_cached",
        "mode": mode,
        "use_llm": True,
        "cases": data.get("cases"),
        "citation_correct_rate": data.get("citation_correct_rate"),
        "false_refusal_rate": data.get("false_refusal_rate"),
        "refusal_accuracy": data.get("refusal_accuracy"),
        "unanswerable_refusal_rate": data.get("unanswerable_refusal_rate"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(retrieval_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> Path:
    out = PROJECT_ROOT / "docs" / "ablation_report.md"
    lines = [
        "# Ablation Report",
        "",
        "This report records low-risk ablations run on the fixed corpus, chunks, BM25 index, and Qdrant dense index. It does not rebuild data or change the production demo configuration.",
        "",
        "## Retrieval Mode Ablation",
        "",
        markdown_table(
            retrieval_rows,
            ["experiment", "doc_hit_at_5", "doc_hit_at_10", "chunk_hit_at_5", "chunk_hit_at_10", "mrr_at_10"],
        ),
        "",
        "## Generation / Refusal Ablation",
        "",
        markdown_table(
            generation_rows,
            [
                "experiment",
                "use_llm",
                "citation_correct_rate",
                "false_refusal_rate",
                "refusal_accuracy",
                "unanswerable_refusal_rate",
            ],
        ),
        "",
        "## Notes",
        "",
        "- Retrieval ablation compares sparse, dense, hybrid fusion, and hybrid plus reranking.",
        "- Generation ablation compares evidence-extractive answering with the cached local-LLM generation evaluation when available.",
        "- Evidence sufficiency remains enabled in all generation rows, because it is part of the safety-critical RAG answer path.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "demo"])
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--with-llm", action="store_true", help="Re-run local LLM generation instead of using cached summary.")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    cases = load_cases(args.split)
    out_dir = Path(load_paths()["data"]["eval"]) / "results"
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        health = client.get(f"{api_base}/health")
        health.raise_for_status()
        retrieval_rows = evaluate_retrieval(client, api_base, cases, args.split, RETRIEVAL_MODES)
        generation_rows = [
            evaluate_generation(
                client,
                api_base,
                cases,
                args.split,
                mode="hybrid_rerank",
                use_llm=False,
                experiment="generation_hybrid_rerank_extractive",
            )
        ]
        if args.with_llm:
            generation_rows.append(
                evaluate_generation(
                    client,
                    api_base,
                    cases,
                    args.split,
                    mode="hybrid_rerank",
                    use_llm=True,
                    experiment="generation_hybrid_rerank_llm",
                )
            )
        else:
            cached = load_cached_llm_summary(args.split, "hybrid_rerank")
            if cached:
                generation_rows.append(cached)

    retrieval_path = out_dir / f"{args.split}_ablation_retrieval.csv"
    generation_path = out_dir / f"{args.split}_ablation_generation.csv"
    write_csv(retrieval_path, retrieval_rows)
    write_csv(generation_path, generation_rows)
    report_path = write_report(retrieval_rows, generation_rows)
    print(retrieval_path)
    print(generation_path)
    print(report_path)


if __name__ == "__main__":
    main()
