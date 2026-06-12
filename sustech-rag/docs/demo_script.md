# Demo Script

## Startup

Terminal 1:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_llm.sh
```

Terminal 2:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_api.sh
```

Terminal 3:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_ui.sh
```

Open Gradio on the mapped AutoDL port for `7860`.

If local requests are tested from Python scripts, set:

```bash
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

## Demonstration Flow

1. Show the data snapshot in `docs/experiment_summary.md`.
   - Current scale: 285 documents, including 243 HTML pages and 42 PDFs; 2649 chunks.
   - Keep `Use local LLM` unchecked for the first pass. This shows faithful extractive answers and avoids small-model wording drift during the defense.
2. Ask a library service question:
   - 南方科技大学图书馆开放时间是什么？
   - Expected: cites the official library page and mentions `周一至周日：8:00 - 22:00` and `一丹一楼24小时开放` when this evidence is retrieved.
3. Ask a PDF curriculum question:
   - 2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？
   - Expected: cites `12-2024级机器人工程专业本科人才培养方案.pdf`.
   - Optional: enable `Use local LLM` on this question to show natural generation. The verified answer is `159学分`.
4. Ask a student-affairs service question:
   - 南方科技大学心理咨询怎么预约？
   - Expected: retrieves Student Affairs mental-health counseling information and shows source cards.
5. Switch retrieval modes:
   - `bm25`
   - `dense`
   - `hybrid`
   - `hybrid_rerank`
   Show the trace section and explain Dense/BM25/RRF/Rerank.
6. Ask an unanswerable question:
   - 南方科技大学是否提供量子传送门预约服务？
   - Expected: refuses due to insufficient official evidence.
7. Show `docs/experiment_summary.md` retrieval and refusal metrics.

## Current Model Note

The current AutoDL runtime has local Qwen2.5-0.5B-Instruct weights available under
`/root/autodl-fs/sustech-rag/models/generator`, so `Use local LLM` can be enabled.
For a safer live demo, leave it unchecked unless explicitly demonstrating generation;
the extractive mode is more faithful for exact facts and still includes citations.
