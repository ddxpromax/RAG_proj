from __future__ import annotations

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

CURATED_QUESTIONS = [
    "南方科技大学图书馆开放时间是什么？",
    "2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？",
    "南方科技大学心理咨询怎么预约？",
    "南方科技大学是否提供量子传送门预约服务？",
]


MODE_CHOICES = [
    ("No-RAG", "no_rag"),
    ("BM25", "bm25"),
    ("Dense", "dense"),
    ("Hybrid", "hybrid"),
    ("Hybrid + Rerank", "hybrid_rerank"),
]


def format_sources(citations: list[dict]) -> str:
    if not citations:
        return "_No cited sources._"
    blocks = []
    for citation in citations[:5]:
        title = citation.get("title") or citation.get("doc_id") or "Untitled"
        url = citation.get("url") or ""
        page = citation.get("page")
        page_text = f" · page {page}" if page else ""
        snippet = " ".join(str(citation.get("snippet") or "").split())
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        source_id = citation.get("source_id")
        blocks.append(f"**[{source_id}] {title}{page_text}**\n\n{url}\n\n> {snippet}")
    return "\n\n---\n\n".join(blocks)


def main() -> None:
    try:
        import gradio as gr
        import httpx
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Gradio and httpx are required for the frontend. Install `.[full]`.") from exc

    from sustech_rag.common.config import load_paths

    paths = load_paths()
    demo_path = Path(paths["data"]["eval"]) / "demo.jsonl"
    demo_questions = list(CURATED_QUESTIONS)
    if demo_path.exists():
        with demo_path.open("r", encoding="utf-8") as f:
            demo_questions.extend(json.loads(line)["question"] for line in f if line.strip())
    demo_questions = list(dict.fromkeys(demo_questions))
    api_base = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8080").rstrip("/")

    def ask(question: str, mode: str, use_llm: bool):
        question = (question or "").strip()
        if not question:
            raise gr.Error("Please enter a question.")
        try:
            with httpx.Client(timeout=120.0, trust_env=False) as client:
                response = client.post(
                    f"{api_base}/chat",
                    json={"question": question, "mode": mode, "use_llm": use_llm},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise gr.Error(f"RAG API request failed: {exc}") from exc
        status = payload.get("evidence_status", "unknown")
        mode_used = payload.get("retrieval_mode", mode)
        answer = f"**Evidence:** `{status}` · **Mode:** `{mode_used}`\n\n{payload.get('answer', '')}"
        sources = format_sources(payload.get("citations") or [])
        return answer, sources, payload.get("trace", {})

    with gr.Blocks(title="SUSTech Campus RAG") as demo:
        gr.Markdown("# SUSTech Campus RAG")
        with gr.Row():
            with gr.Column(scale=2):
                sample = gr.Dropdown(demo_questions, label="Demo question", value=demo_questions[0])
                question = gr.Textbox(label="Question", lines=3, value=demo_questions[0] if demo_questions else "")
            with gr.Column():
                mode = gr.Radio(
                    MODE_CHOICES,
                    value="hybrid_rerank",
                    label="Retrieval mode",
                )
                use_llm = gr.Checkbox(value=True, label="Use local LLM")
                submit = gr.Button("Ask")
        with gr.Tabs():
            with gr.Tab("Answer"):
                answer = gr.Markdown()
            with gr.Tab("Sources"):
                citations = gr.Markdown()
            with gr.Tab("Trace"):
                trace = gr.JSON()
        sample.change(lambda value: value or "", inputs=sample, outputs=question)
        submit.click(ask, inputs=[question, mode, use_llm], outputs=[answer, citations, trace])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))


if __name__ == "__main__":
    main()
