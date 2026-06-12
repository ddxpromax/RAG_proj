from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import load_paths
from sustech_rag.generation.rag import RAGService


def load_cases(split: str) -> list[dict]:
    path = Path(load_paths()["data"]["eval"]) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_case(case: dict, response) -> dict:
    answer = response.answer
    citations = response.citations
    answerable = bool(case.get("answerable", True))
    relevant_docs = set(case.get("relevant_doc_ids") or [])
    relevant_chunks = set(case.get("relevant_chunk_ids") or [])
    citation_ids = {c.get("chunk_id") for c in citations} | {c.get("doc_id") for c in citations}
    citation_correct = bool((relevant_docs | relevant_chunks) & citation_ids) if answerable else True
    refused = response.evidence_status == "insufficient_evidence" or "无法根据" in answer
    required_text = "".join(case.get("required_facts") or [])
    fact_overlap = 0
    for token in required_text[:80].split():
        if token and token in answer:
            fact_overlap += 1
    return {
        "question_id": case["question_id"],
        "answerable": answerable,
        "evidence_status": response.evidence_status,
        "has_citation": bool(citations),
        "citation_correct": citation_correct,
        "refused": refused,
        "refusal_correct": (not answerable and refused) or (answerable and not refused),
        "answer_preview": answer[:160].replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "demo"], default="test")
    parser.add_argument("--mode", default="hybrid_rerank")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.split)
    service = RAGService()
    rows = []
    for case in cases:
        response = service.answer(case["question"], mode=args.mode, use_llm=args.use_llm)
        row = score_case(case, response)
        row["question"] = case["question"]
        row["mode"] = args.mode
        rows.append(row)

    out_dir = Path(load_paths()["data"]["eval"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_generation_{args.mode}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    summary = {
        "split": args.split,
        "mode": args.mode,
        "cases": len(rows),
        "citation_correct_rate": sum(r["citation_correct"] for r in answerable_rows) / max(1, len(answerable_rows)),
        "false_refusal_rate": sum(r["refused"] for r in answerable_rows) / max(1, len(answerable_rows)),
        "refusal_accuracy": sum(r["refusal_correct"] for r in rows) / max(1, len(rows)),
        "unanswerable_refusal_rate": sum(r["refused"] for r in unanswerable_rows) / max(1, len(unanswerable_rows)),
    }
    summary_path = out_dir / f"{args.split}_generation_{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
