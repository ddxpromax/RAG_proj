# Presentation Notes

## 1. Five-Minute Talk Track

### Opening

大家好，我的项目是一个面向南方科技大学公开校园信息的 RAG 问答系统。它对应项目一：构建知识库并实现基于检索增强生成的问答。

我选择这个方向的原因是，学校信息分散在主站、教学工作部、学生工作部、图书馆、研究生院、国际合作网站以及公开 PDF 中。直接让大模型回答容易过时或者幻觉，所以这个系统的核心目标是：先从官方资料检索证据，再基于证据回答，并在证据不足时拒答。

### Data

数据全部来自公开官方来源。当前版本有 285 篇规范化文档，其中 243 篇 HTML、42 篇 PDF，总共切分成 2649 个 chunks。

PDF 部分主要是 2024 级本科人才培养方案。系统会自动发现 PDF、下载、解析，并把来源 URL 和页码信息保留到 citation 里。

### Method

系统流程是：

```text
source registry -> crawl/PDF fetch -> parse -> chunk -> BM25/Dense index -> hybrid retrieval -> rerank -> evidence check -> generation
```

检索部分实现了四种模式：

- BM25：适合关键词精确匹配。
- Dense：使用 Qwen3-Embedding-0.6B 做语义向量检索，向量存到本地 Qdrant。
- Hybrid：把 BM25 和 Dense 用 RRF 融合。
- Hybrid + Rerank：再用 Qwen3-Reranker-0.6B 做精排。

生成模型是本地 Qwen2.5-0.5B-Instruct，通过 OpenAI-compatible 接口服务在 8000 端口。API 在 8080，Gradio UI 在 7860。

### Trustworthiness

我重点做了两个可信机制：

第一，所有回答都带来源引用，可以看到标题、URL 和片段。

第二，系统有 evidence sufficiency check。如果问题里包含资料中没有的具体限定，比如不存在的“火星校区”或者“量子传送门预约服务”，系统会拒答，而不是编答案。

### Results

在 test 集上，Dense、BM25、Hybrid、Hybrid+Rerank 都做了评估。

最重要的结果是：

- Hybrid+Rerank 的 `doc_hit@10` 是 0.967，`MRR@10` 是 0.876。
- 生成评测中 citation correct rate 是 0.917。
- false refusal rate 是 0。
- refusal accuracy 是 1.0。
- unanswerable refusal rate 是 1.0。

这说明系统能比较稳定地找到官方证据，并且对不可回答问题保持拒答。

此外我补充做了消融实验：

- BM25 的 chunk hit 更高，说明精确词匹配对学分、邮箱、电话这类事实很有效。
- Dense 的 doc hit 更高，说明语义召回更容易找到正确文档。
- Hybrid 的 MRR 最好，说明二者融合后排序更稳。
- 当前 0.6B reranker 比较保守，没有在 test 集显著提升 doc@5，所以报告如实保留了这个 tradeoff。
- reranker 权重消融显示，越依赖 Qwen reranker，chunk 命中略有提升，但 MRR 会下降；所以最终采用“Qwen 分数 + 原 hybrid rank prior”的保守融合。
- evidence gate 消融显示，如果强制回答、不做证据充足性判断，不可回答题拒答率会从 1.0 掉到 0，所以拒答门控是必要的。

### Demo Transition

接下来我展示三个问题：

1. 图书馆开放时间，展示普通网页检索。
2. 机器人工程培养方案最低学分，展示 PDF 检索和引用。
3. 量子传送门预约服务，展示拒答机制。

如果老师问 No-RAG 对比，可以切到 `No-RAG` 模式说明：不加 RAG 时模型只依赖参数知识，无法保证知道最新 PDF 或官方页面；加 RAG 后能展示来源、证据片段和拒答状态。

## 2. Demo Flow

Open:

```text
http://127.0.0.1:7860
```

Keep settings:

```text
Retrieval mode: Hybrid + Rerank
Use local LLM: unchecked for the main demo
```

Use the extractive mode for the first pass because it quotes the retrieved evidence
directly and is safest for exact facts. If you want to show natural generation,
enable `Use local LLM` only for the PDF curriculum question after the cited evidence
has already been shown.

### Demo 1: Library

Question:

```text
南方科技大学图书馆开放时间是什么？
```

What to show:

- Evidence status should be `supported`.
- Sources should include official library pages.
- Mention that the system can show source snippets and trace.

Suggested explanation:

这个问题来自网页类资料。系统先检索图书馆页面，再把证据传给本地 LLM 生成回答，并保留 citation。

### Demo 2: PDF Curriculum

Question:

```text
2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？
```

Expected answer:

```text
159 学分
```

What to show:

- Citation title should include `12-2024级机器人工程专业本科人才培养方案`.
- This demonstrates PDF ingestion and retrieval.

Suggested explanation:

这个问题来自 PDF 培养方案。项目不需要手动上传 PDF，脚本会从公开目录发现和下载 PDF，然后解析、切块、建立索引。

Optional LLM generation line:

```text
如果勾选 Use local LLM，本地 Qwen2.5-0.5B 会把证据整理成自然语言回答；这个问题已验证会回答 159 学分。
```

### Demo 3: Student Affairs

Question:

```text
南方科技大学心理咨询怎么预约？
```

What to show:

- Sources should include Student Affairs / psychology counseling pages.
- This demonstrates another category beyond teaching and library data.

Suggested explanation:

这个问题展示系统不是只围绕一个网站，而是整合多个部门公开信息。

### Demo 4: Refusal

Question:

```text
南方科技大学是否提供量子传送门预约服务？
```

Expected answer:

```text
当前知识库没有检索到足够直接证据，无法根据现有官方资料回答该问题。
```

Suggested explanation:

这里不是检索不到任何内容，而是检索结果不能支持问题里的具体事实，所以 evidence status 是 insufficient_evidence。这个机制用来降低幻觉风险。

## 3. Key Numbers to Memorize

| item | value |
| --- | ---: |
| documents | 285 |
| HTML documents | 243 |
| PDF documents | 42 |
| chunks | 2649 |
| test Hybrid+Rerank doc_hit@10 | 0.967 |
| test Hybrid+Rerank MRR@10 | 0.876 |
| citation correct rate | 0.917 |
| false refusal rate | 0.000 |
| refusal accuracy | 1.000 |
| unanswerable refusal rate | 1.000 |

## 4. Likely Questions and Answers

### Q1: 为什么不用纯大模型直接回答？

纯大模型可能不知道最新校园信息，也可能编造。RAG 的好处是回答前先检索官方资料，结果可追踪、可引用，也能对证据不足的问题拒答。

本项目也保留了 `No-RAG` 模式作为 baseline，用来现场对比“直接问模型”和“先检索官方证据再回答”的区别。

### Q2: 为什么要同时做 BM25 和 Dense？

BM25 对关键词、年份、专业名称这类精确匹配很强；Dense 对语义相近但措辞不同的问题更强。Hybrid 用 RRF 融合两者，整体更稳。

### Q3: 为什么 reranker 没有让所有 test 指标都提升？

当前 reranker 是 0.6B 小模型，并且数据集较小、问题类型较混合。它在 dev 集 MRR 有提升，但 test 集 doc@5 略降。项目保留所有模式并如实报告这个 tradeoff，说明系统评估不是只挑最好看的数字。

从消融看，Hybrid 本身已经很强，reranker 主要保持 doc@10 和 chunk 命中，没有稳定带来前 5 位收益。后续可以用更大的 reranker 或人工标注数据做校准。

权重消融进一步说明：Qwen-only rerank 的 doc@10 可以到 0.983，但 MRR 降到 0.825，说明它能找到相关文档，却不一定把最合适证据排在最前。

### Q4: 为什么选择 Qwen2.5-0.5B 作为生成模型？

主要考虑 AutoDL 环境和显存稳定性。系统配置里保留了更大模型目标，但最终演示优先保证可本地运行、可复现、延迟可接受。

### Q5: PDF 是怎么处理的？

`scripts/fetch_pdf_sources.py` 从配置的公开目录发现 PDF，下载后写入 raw manifest。`parse.py` 用 PyMuPDF 解析文本并保留页码标记，之后和 HTML 一样进入 chunk、BM25、embedding 和 Qdrant。

### Q6: 怎么判断应该拒答？

系统会检查检索证据是否覆盖问题中的关键限定，例如年份、校区名、专业名等。对于明显不存在或资料不支持的问题，返回 `insufficient_evidence`，避免模型自由发挥。

消融结果也支持这个设计：关闭 evidence gate 并强制回答时，不可回答问题的拒答率从 1.0 下降到 0，说明仅有检索结果不等于证据足够。

### Q7: 为什么前端不直接读取索引？

因为当前使用 embedded Qdrant，本地文件有锁。Gradio UI 通过 API 调用，让 Qdrant 只由 API 进程访问，演示时更稳定。

### Q8: 如何复现完整实验？

运行 `docs/operations.md` 里的 rebuild commands。核心顺序是 crawl/PDF fetch、parse、chunk、BM25、embedding、generate_eval、evaluate、evaluate_generation、report_summary。

## 5. If Something Goes Wrong

### API health failed

Run:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_api.sh
```

### Local LLM health failed

Run:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_llm.sh
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

### UI failed

Run:

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_ui.sh
```

### Localhost request returns 502

Set:

```bash
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

### Final health check

Run:

```bash
python scripts/health_check.py
```

Expected result:

```text
OK documents 285
OK chunks 2649
OK hybrid_rerank_query via_api hits=5
OK local_llm_endpoint backend=transformers
```
