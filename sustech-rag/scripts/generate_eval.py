from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.chunking.chunker import load_chunks
from sustech_rag.common.config import load_paths


TYPE_BY_CATEGORY = {
    "undergraduate_teaching": "undergraduate_rule",
    "student_affairs": "campus_service",
    "library": "campus_service",
    "global": "international",
    "university_overview": "single_fact",
    "admissions": "admissions",
    "graduate": "graduate",
}


def question_for(chunk: dict) -> str:
    meta = chunk["metadata"]
    title = meta.get("title") or "该资料"
    category = meta.get("category")
    text = chunk["text"]
    phrase = key_phrase(text)
    if category == "library":
        return f"根据图书馆官方资料，{title} 中“{phrase}”相关内容是什么？"
    if category == "undergraduate_teaching":
        return f"根据教学工作部资料，{title} 中“{phrase}”相关内容是什么？"
    if category == "student_affairs":
        return f"根据学生工作部资料，{title} 中“{phrase}”涉及哪些学生事务信息？"
    if category == "global":
        return f"根据国际合作部资料，{title} 中“{phrase}”介绍了什么项目或服务？"
    if category == "admissions":
        return f"根据本科招生资料，{title} 中“{phrase}”相关规定是什么？"
    if category == "graduate":
        return f"根据研究生院资料，{title} 中“{phrase}”相关内容是什么？"
    if "学分" in text or "培养方案" in text:
        return f"{title} 中“{phrase}”关于培养或学分的核心信息是什么？"
    return f"南方科技大学官方资料《{title}》中“{phrase}”的核心内容是什么？"


def key_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    for part in re.split(r"[。！？；\n]", cleaned):
        part = part.strip(" -|")
        if 8 <= len(part) <= 36 and not part.startswith(("首页", "当前位置")):
            return part[:28]
    return cleaned[:28]


def is_eval_candidate(chunk) -> bool:
    text = re.sub(r"\s+", " ", chunk.display_text).strip()
    if len(text) < 120:
        return False
    if any(bad in text for bad in ["登录", "CAS", "版权所有"]):
        return False
    nav_terms = [
        "首页",
        "部门概况",
        "部门介绍",
        "机构设置",
        "部门领导",
        "新闻中心",
        "学业信息",
        "教务系统",
        "就业指导",
        "学生事务",
        "奖助贷保",
        "心理健康",
    ]
    if sum(1 for term in nav_terms if term in text[:220]) >= 5:
        return False
    if text.startswith(("首页 ", "当前位置", "English ")) and "。" not in text[:160]:
        return False
    title = str(chunk.metadata.get("title") or "")
    if title.startswith("- ") or title in {"学生工作部", "网络资源导航"}:
        return False
    phrase = key_phrase(text)
    if sum(1 for term in nav_terms if term in phrase) >= 3:
        return False
    return True


def build_case(chunk, index: int) -> dict:
    meta = chunk.metadata
    return {
        "question_id": f"q{index:03d}",
        "question": question_for(chunk.model_dump()),
        "type": TYPE_BY_CATEGORY.get(meta.get("category"), "single_fact"),
        "reference_answer": chunk.display_text[:260],
        "required_facts": [chunk.display_text[:80]],
        "relevant_doc_ids": [chunk.doc_id],
        "relevant_chunk_ids": [chunk.chunk_id],
        "answerable": True,
        "effective_year": meta.get("effective_year"),
        "source_title": meta.get("title"),
        "source_url": meta.get("url"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--dev", type=int, default=15)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--demo", type=int, default=10)
    args = parser.parse_args()
    random.seed(args.seed)
    chunks = [c for c in load_chunks() if is_eval_candidate(c)]
    by_category: dict[str, list] = {}
    for chunk in chunks:
        by_category.setdefault(chunk.metadata.get("category", "unknown"), []).append(chunk)
    selected = []
    target_total = args.dev + args.test + args.demo
    categories = [
        "undergraduate_teaching",
        "admissions",
        "graduate",
        "student_affairs",
        "library",
        "global",
        "university_overview",
    ]
    while len(selected) < target_total and any(by_category.get(c) for c in categories):
        for category in categories:
            bucket = by_category.get(category) or []
            if bucket and len(selected) < target_total:
                selected.append(bucket.pop(random.randrange(len(bucket))))
    cases = [build_case(chunk, i + 1) for i, chunk in enumerate(selected)]
    no_answer_questions = [
        "南方科技大学火星校区食堂几点开放？",
        "南方科技大学是否提供量子传送门预约服务？",
        "2029级所有专业的培养方案最低毕业学分是多少？",
        "南方科技大学校内私人银行贷款利率是多少？",
        "南方科技大学尚未公开的下学期考试答案在哪里下载？",
    ]
    for question in no_answer_questions:
        cases.append(
            {
                "question_id": f"q{len(cases)+1:03d}",
                "question": question,
                "type": "unanswerable",
                "reference_answer": "当前公开官方资料不足，系统应拒答。",
                "required_facts": [],
                "relevant_doc_ids": [],
                "relevant_chunk_ids": [],
                "answerable": False,
                "effective_year": None,
            }
        )
    paths = load_paths()
    out_dir = Path(paths["data"]["eval"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "dev.jsonl": cases[: args.dev],
        "test.jsonl": cases[args.dev : args.dev + args.test],
        "demo.jsonl": cases[args.dev + args.test : args.dev + args.test + args.demo],
    }
    # Include a few unanswerable cases in test even when generated after the answerable slice.
    splits["test.jsonl"].extend(cases[-len(no_answer_questions) :])
    for filename, rows in splits.items():
        with (out_dir / filename).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(filename, len(rows))


if __name__ == "__main__":
    main()
