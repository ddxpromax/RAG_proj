from __future__ import annotations

import re
import uuid

from sustech_rag.common.config import PROJECT_ROOT, load_yaml
from sustech_rag.common.schema import ChatResponse, RetrievalHit
from sustech_rag.generation.client import LLMClient
from sustech_rag.retrieval.service import RetrievalService


def build_context(hits: list[RetrievalHit]) -> tuple[str, list[dict]]:
    blocks = []
    citations = []
    for idx, hit in enumerate(hits, start=1):
        page = hit.metadata.get("page_start") or hit.metadata.get("page")
        title = hit.title or hit.metadata.get("title") or hit.doc_id
        blocks.append(
            f"[{idx}]\n标题：{title}\n适用年份：{hit.metadata.get('effective_year') or '未知'}\n"
            f"来源：{hit.url or hit.metadata.get('url') or 'unknown'}\n内容：{hit.text}"
        )
        citations.append(
            {
                "source_id": idx,
                "doc_id": hit.doc_id,
                "chunk_id": hit.chunk_id,
                "title": title,
                "url": str(hit.url or hit.metadata.get("url") or ""),
                "page": page,
                "snippet": hit.text[:240],
            }
        )
    return "\n\n".join(blocks), citations


class RAGService:
    def __init__(self, retrieval: RetrievalService | None = None) -> None:
        self.retrieval = retrieval or RetrievalService()
        self.generation_config = load_yaml(PROJECT_ROOT / "configs" / "generation.yaml")

    def answer(self, query: str, mode: str = "bm25", use_llm: bool = True) -> ChatResponse:
        trace_id = str(uuid.uuid4())
        if mode == "no_rag":
            if use_llm:
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个中文问答助手。直接回答用户问题，不使用外部检索资料。",
                    },
                    {"role": "user", "content": query},
                ]
                answer = LLMClient().chat(messages)
            else:
                answer = "No-RAG baseline skipped retrieval. Enable use_llm after vLLM is running."
            return ChatResponse(
                answer=answer,
                evidence_status="insufficient_evidence",
                trace_id=trace_id,
                retrieval_mode=mode,
            )
        retrieval = self.retrieval.retrieve(query, mode=mode)
        hits = retrieval["hits"]
        context, citations = build_context(hits)
        evidence_status = self.evidence_status(query, hits)
        if not hits or evidence_status == "insufficient_evidence":
            return ChatResponse(
                answer="当前知识库没有检索到足够直接证据，无法根据现有官方资料回答该问题。",
                evidence_status="insufficient_evidence",
                citations=citations,
                trace_id=trace_id,
                retrieval_mode=mode,
                trace=retrieval["trace"],
            )
        if not use_llm:
            retrieval["trace"]["llm_used"] = False
            answer = self.extractive_answer(query, hits)
        else:
            messages = [
                {"role": "system", "content": self.generation_config["system_prompt"]},
                {"role": "user", "content": f"问题：{query}\n\n官方资料：\n{context}"},
            ]
            try:
                answer = LLMClient().chat(messages)
                retrieval["trace"]["llm_used"] = True
            except Exception as exc:
                retrieval["trace"]["llm_error"] = str(exc)
                retrieval["trace"]["llm_used"] = False
                answer = self.extractive_answer(query, hits)
        return ChatResponse(
            answer=answer,
            evidence_status=evidence_status,
            citations=citations,
            trace_id=trace_id,
            retrieval_mode=mode,
            trace=retrieval["trace"],
        )

    @staticmethod
    def extractive_answer(query: str, hits) -> str:
        if not hits:
            return "当前知识库没有检索到足够证据，无法根据官方资料回答。"
        lines = ["根据当前检索到的官方资料："]
        top_k = 1 if RAGService.is_precise_fact_query(query) else 3
        for idx, hit in enumerate(hits[:top_k], start=1):
            snippet = RAGService.best_snippet(query, hit.text)
            lines.append(f"{idx}. {snippet} [{idx}]")
        lines.append("以上为证据摘录式回答。")
        return "\n".join(lines)

    @staticmethod
    def is_precise_fact_query(query: str) -> bool:
        precise_markers = [
            ("学分", "最低"),
            ("开放", "时间"),
            ("心理咨询", "预约"),
            ("电话", "多少"),
            ("邮箱", "什么"),
        ]
        return any(all(marker in query for marker in markers) for markers in precise_markers)

    @staticmethod
    def best_snippet(query: str, text: str, limit: int = 220) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        priority = RAGService.priority_snippet(query, cleaned, limit)
        if priority:
            return priority
        terms = RAGService.important_terms(query)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*", cleaned) if len(s.strip()) >= 12]
        if not sentences:
            return cleaned[:limit] + ("..." if len(cleaned) > limit else "")
        scored = []
        for index, sentence in enumerate(sentences[:40]):
            score = sum(2 for term in terms if term in sentence)
            score += sum(
                1
                for key in ["开放", "时间", "招生", "章程", "专业", "学分", "申请", "办理"]
                if key in query and key in sentence
            )
            score -= index * 0.02
            scored.append((score, sentence))
        snippet = " ".join(sentence for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:2]).strip()
        if not snippet:
            snippet = cleaned
        return snippet[:limit] + ("..." if len(snippet) > limit else "")

    @staticmethod
    def priority_snippet(query: str, text: str, limit: int) -> str:
        patterns: list[str] = []
        if "学分" in query:
            patterns.extend(
                [
                    r"[^。；;]*最低学分要求[^。；;]*?为\s*\d+\s*学分[^。；;]*[。；;]?",
                    r"[^。；;]*毕业最低学分要求[^。；;]*?\d+\s*学分[^。；;]*[。；;]?",
                ]
            )
        if "开放" in query and "时间" in query:
            patterns.extend(
                [
                    r"[^。；;]*周一至周日[^。；;]*?\d{1,2}[:：.]\d{2}\s*[-—至]\s*\d{1,2}[:：.]\d{2}[^。；;]*[。；;]?",
                    r"[^。；;]*24小时开放[^。；;]*[。；;]?",
                ]
            )
        if "心理咨询" in query or "预约" in query:
            patterns.extend(
                [
                    r"[^。；;]*邮箱预约[^。；;]*?counseling@sustc\.edu\.cn[^。；;]*[。；;]?",
                    r"[^。；;]*电话预约[^。；;]*?88010576[^。；;]*[。；;]?",
                ]
            )
        snippets = []
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                snippet = match.group(0).strip()
                if snippet and snippet not in snippets:
                    snippets.append(snippet)
        if not snippets:
            return ""
        combined = " ".join(snippets)
        return combined[:limit] + ("..." if len(combined) > limit else "")

    @staticmethod
    def evidence_status(query: str, hits) -> str:
        if not hits:
            return "insufficient_evidence"
        combined = "\n".join((hit.title or "") + "\n" + hit.text for hit in hits[:5])
        query_years = set(re.findall(r"20[0-3]\d", query))
        if query_years and not any(year in combined or str(hit.metadata.get("effective_year")) in query_years for hit in hits[:5] for year in query_years):
            return "insufficient_evidence"

        for qualifier in RAGService.required_qualifiers(query):
            if qualifier not in combined:
                return "insufficient_evidence"

        terms = RAGService.important_terms(query)
        if not terms:
            return "supported"
        matched = sum(1 for term in terms if term in combined)
        unmatched_specific = [term for term in terms if term not in combined and term not in RAGService.generic_terms()]
        if len(unmatched_specific) >= 2:
            return "insufficient_evidence"
        coverage = matched / max(1, len(terms))
        if coverage >= 0.5:
            return "supported"
        if coverage >= 0.3 and hits[0].score > 40:
            return "partially_supported"
        return "insufficient_evidence"

    @staticmethod
    def required_qualifiers(query: str) -> list[str]:
        qualifiers = []
        prefixes = ("南方科技大学", "南科大")
        for match in re.finditer(r"([\u4e00-\u9fff]{2,12})校区", query):
            qualifier = match.group(1)
            for prefix in prefixes:
                if qualifier.startswith(prefix):
                    qualifier = qualifier[len(prefix) :]
            qualifier = qualifier.strip()
            if qualifier and qualifier not in {"大学", "科技大学"}:
                qualifiers.append(qualifier)
        return list(dict.fromkeys(qualifiers))

    @staticmethod
    def important_terms(query: str) -> list[str]:
        stop = {
            "南方科技大学",
            "南科大",
            "什么",
            "哪些",
            "有关",
            "相关",
            "根据",
            "官方",
            "资料",
            "怎么",
            "是否",
            "多少",
            "要点",
            "内容",
            "南方",
            "科技",
            "大学",
            "提供",
            "服务",
            "预约",
            "开放",
            "时间",
            "几点",
            "教学",
            "工作部",
            "本科",
            "专业本科",
            "本专业",
            "人才培养",
            "培养方案",
            "最低",
            "要求",
        }
        try:
            import jieba

            terms = [term.strip() for term in jieba.cut(query) if len(term.strip()) >= 2]
        except Exception:
            terms = re.findall(r"[A-Za-z]{2,}\d*|20[0-3]\d|[\u4e00-\u9fff]{2,}", query)
        cleaned = []
        for term in terms:
            if term in stop:
                continue
            if re.fullmatch(r"[？?，,。；;的了在是和与中]", term):
                continue
            cleaned.append(term)
        return list(dict.fromkeys(cleaned[:8]))

    @staticmethod
    def generic_terms() -> set[str]:
        return {
            "南方",
            "科技",
            "大学",
            "提供",
            "服务",
            "预约",
            "开放",
            "时间",
            "几点",
            "校区",
            "相关",
            "内容",
        }
