# SUSTech RAG Defense Presentation Guide

这份文档按正式答辩展示顺序组织，可以作为“文档版 PPT”使用。内容覆盖项目目标、系统流程、技术实现、实验结果、现场演示和常见追问，适合答辩时直接顺序讲解。

## 0. 开场定位

大家好，我的项目是一个面向南方科技大学公开校园信息的 RAG 问答系统。这个项目对应课程项目一，也就是构建知识库，并基于检索增强生成实现问答。

我选择这个题目的原因是，学校相关信息本身比较分散：有主站、教学工作部、学生工作部、图书馆、研究生院、国际合作相关网站，还有公开的本科培养方案 PDF。对于这类信息，如果直接让大语言模型回答，它可能不知道最新内容，也可能根据已有参数知识进行猜测。因此这个项目的核心目标不是单纯让模型“说得像”，而是让系统先检索官方资料，再基于检索到的证据回答，并且在证据不足时拒答。

最终系统覆盖了从公开数据获取、解析清洗、chunk 切分、稀疏与稠密索引、混合检索、重排序、证据充足性判断、本地生成、API 服务、Gradio 前端，到实验评估和消融分析的完整流程。

本项目最终实现的是一个可以本地运行、可以展示引用来源、可以比较不同检索策略、并且具备拒答机制的校园知识问答系统。

关键要点：

- 项目不是手工写固定答案，而是从公开官方网页和 PDF 中构建检索知识库。
- 生成模型是本地 `Qwen2.5-0.5B-Instruct`，通过 OpenAI-compatible 接口调用。
- Embedding 使用 `Qwen3-Embedding-0.6B`。
- Reranker 使用 `Qwen3-Reranker-0.6B`。
- 系统默认推荐 `hybrid_rerank` 检索模式，但保留了 BM25、Dense、Hybrid、No-RAG 等对比模式。

## 1. 项目目标与任务完成情况

这个项目的目标可以拆成六个部分。

第一，收集南方科技大学公开官方资料，形成一个可复现的数据来源清单。我的数据来源不是临时复制粘贴，而是写在配置文件中，由脚本统一爬取或下载。

第二，对网页和 PDF 进行解析、清洗和规范化。HTML 页面需要抽取正文、标题、URL 和类别；PDF 需要解析文本，并保留页码和文件来源，方便后续 citation。

第三，把文档切分成适合检索的 chunks。因为单篇网页或 PDF 通常太长，直接送入模型会超出上下文，也会降低检索粒度，所以需要切块并保留 overlap。

第四，建立不同类型的索引，包括 BM25 稀疏检索和 dense vector 稠密检索。这样可以同时覆盖关键词精确匹配和语义匹配。

第五，实现 RAG 问答链路。用户输入问题后，系统先检索相关 chunks，再进行混合融合、rerank、证据判断，最后生成或抽取回答，并返回 citations。

第六，进行实验评估和消融分析。我不仅给出最终系统效果，也比较了 BM25、Dense、Hybrid、Hybrid+Rerank，不同 reranker 权重，以及 evidence gate 开关对拒答能力的影响。

因此，从任务要求角度看，本项目已经完成了数据获取、知识库构建、检索增强问答、No-RAG 对比、实验评估、前端展示和可复现文档。

可以展示的项目结构：

```text
sustech-rag/
├── configs/          # 数据源、模型、路径、检索和生成配置
├── scripts/          # 爬取、解析、切块、建索引、评估和消融脚本
├── src/              # RAG 核心代码
├── frontend/         # Gradio 前端
├── tests/            # 单元测试
└── docs/             # 报告、实验、答辩和操作文档
```

## 2. 数据来源与数据规模

本项目的数据全部来自公开官方来源，来源注册在 `configs/sources.yaml` 中。主要包括南方科技大学主站、教学工作部、学生工作部、图书馆、研究生院、国际合作相关网站，以及公开的 2024 级本科培养方案 PDF。

当前数据快照中，一共有 285 篇规范化文档，其中 243 篇来自 HTML 页面，42 篇来自 PDF。经过切分后得到 2649 个 chunks。

这里有两个细节比较重要。

第一个细节是 PDF 并不是手动放进项目里的。项目中有 `scripts/fetch_pdf_sources.py`，会从配置的公开目录中发现并下载 PDF。这样做的好处是数据来源可追踪，后续也可以重新构建。

第二个细节是 PDF 解析时保留了页码和标题信息。比如 2024 级机器人工程专业本科人才培养方案，系统可以在 citation 中显示对应 PDF 标题，而不是只显示一个难以理解的文件编号。

数据规模表：

| item | value |
| --- | ---: |
| normalized documents | 285 |
| HTML documents | 243 |
| PDF documents | 42 |
| chunks | 2649 |
| embedding model | Qwen3-Embedding-0.6B |
| reranker model | Qwen3-Reranker-0.6B |
| generator model | Qwen2.5-0.5B-Instruct |

关键要点：

- 大文件、模型和索引不直接提交到 GitHub，而是存放在 AutoDL 挂载目录 `/root/autodl-fs/sustech-rag`。
- GitHub 仓库保留代码、配置、脚本和文档，保证结构清晰。
- 运行时数据包括 raw HTML、raw PDF、normalized JSONL、chunks、BM25 artifact、Qdrant 向量库和评估输出。

## 3. 数据处理流水线

整个数据处理流水线可以概括为：

```text
source registry
-> crawl / PDF fetch
-> raw manifest
-> parse
-> normalize
-> chunk
-> BM25 index and dense vector index
-> retrieve
-> rerank
-> evidence check
-> answer generation
```

首先，数据源由配置文件管理。HTML 页面通过 crawler 获取，PDF 通过 PDF fetch 脚本从公开目录发现和下载。下载后的原始文件会有 manifest 记录来源。

然后进入 parse 阶段。HTML 会抽取正文和元信息，PDF 使用 PyMuPDF 解析文本并保留页码。解析后的文档会统一转成 normalized document 格式，这样后续 chunk 和 index 不需要区分它原来是网页还是 PDF。

接着进入 chunk 阶段。chunk 的作用是把长文档拆成更适合检索和放入上下文的片段。系统会尽量保留页面信息和重叠窗口，这样一个事实如果跨越边界，也不容易被切断。

最后是索引阶段。BM25 索引用于关键词检索，dense index 使用 embedding 模型把 chunk 编码成向量并写入本地 Qdrant。到这里，知识库就可以支持多种检索模式。

关键脚本：

| script | purpose |
| --- | --- |
| `scripts/crawl.py` | 爬取配置中的网页来源 |
| `scripts/fetch_pdf_sources.py` | 发现并下载公开 PDF |
| `scripts/parse.py` | 解析 HTML/PDF 并规范化 |
| `scripts/chunk.py` | 文档切块 |
| `scripts/build_bm25.py` | 构建 BM25 稀疏索引 |
| `scripts/embed.py` | 构建 Qdrant 稠密向量索引 |
| `scripts/generate_eval.py` | 生成 dev/test/demo 评测集 |
| `scripts/evaluate.py` | 检索评估 |
| `scripts/evaluate_generation.py` | 生成与拒答评估 |
| `scripts/run_ablations.py` | 消融实验 |

## 4. 检索架构设计

检索部分是这个 RAG 系统的核心。我实现了四种主要检索模式：BM25、Dense、Hybrid、Hybrid+Rerank。

BM25 是稀疏关键词检索。它的优势是对精确词非常敏感，比如专业名称、年份、电话、邮箱、学分这些信息。对于校园问答来说，这类精确信息很多，所以 BM25 是必要的。

Dense retrieval 是稠密语义检索。它使用 `Qwen3-Embedding-0.6B` 把 query 和 chunks 编码成向量，通过向量相似度搜索相关内容。它的优势是当用户问题和原文措辞不完全一致时，仍然可能找到语义相关的文档。

Hybrid retrieval 是把 BM25 和 Dense 的结果进行融合。本项目使用 Reciprocal Rank Fusion，也就是 RRF。它不直接比较两种检索分数，因为 BM25 分数和向量相似度不在同一个尺度上，而是根据排名进行融合。RRF 的基本思想是：一个 chunk 在多个检索器中排名越靠前，融合后的得分越高。

公式可以简单理解为：

```text
score(chunk) = sum(1 / (k + rank_i))
```

其中 `rank_i` 是该 chunk 在某个检索器结果中的排名，`k` 在当前配置中是 60。

在 Hybrid 之后，系统还支持 Hybrid+Rerank。Reranker 的作用是对前面召回的一批候选 chunks 进行更精细的相关性判断。当前使用 `Qwen3-Reranker-0.6B`，通过 yes/no causal scoring 的方式判断文档是否满足 query 需求。

检索配置：

| item | value |
| --- | ---: |
| BM25 top-k | 30 |
| Dense top-k | 30 |
| RRF k | 60 |
| Fusion top-k | 30 |
| Rerank top-k | 30 |
| Context top-k | 5 |
| Per-document limit | 3 |

关键要点：

- BM25 与 Dense 都先召回 30 个候选。
- Hybrid 通过 RRF 得到融合候选。
- Rerank 对融合后的候选重新排序。
- 最终进入回答上下文的是去重后的 top 5。
- `per_doc_limit=3` 用来避免同一文档占满上下文。

## 5. Reranker 的具体实现

Reranker 这一层使用的是 Qwen 的 yes/no 相关性判断思路。对于每一个候选 chunk，系统会构造一个包含 instruction、query 和 document 的输入，让模型判断这个 document 是否满足 query 的需求，答案只能是 yes 或 no。

实现上，系统取模型在最后位置对 `yes` 和 `no` 两个 token 的概率，然后把 `yes` 的概率作为 Qwen rerank score。这个分数代表模型认为该 chunk 与问题相关的程度。

但是我没有完全相信 reranker 分数，而是把它和原始 hybrid rank prior 做了融合。原因是当前 reranker 是 0.6B 小模型，而且课程项目数据集不大，如果完全依赖 reranker，可能会把本来排序很好的证据挪到后面。

最终分数可以理解为：

```text
final_score = qwen_score_weight * qwen_score
            + (1 - qwen_score_weight) * rank_prior
```

当前生产配置使用的是比较保守的融合权重，也就是让 reranker 提供辅助判断，但不完全覆盖原有 Hybrid 排名。

这一点也通过消融实验验证了。随着 Qwen reranker 权重上升，chunk hit 有轻微提升，但 MRR 下降，说明 reranker 能识别一些相关片段，却不一定总是把最优证据排在第一位。

Reranker 权重消融：

| qwen_score_weight | doc_hit@5 | doc_hit@10 | chunk_hit@5 | MRR@10 | interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.967 | 0.967 | 0.817 | 0.877 | original hybrid rank only |
| 0.25 | 0.917 | 0.967 | 0.817 | 0.876 | current conservative blend |
| 0.50 | 0.917 | 0.967 | 0.833 | 0.875 | slightly better chunk hit |
| 0.75 | 0.917 | 0.967 | 0.833 | 0.874 | similar chunk hit, lower MRR |
| 1.00 | 0.900 | 0.983 | 0.833 | 0.825 | Qwen-only over-shifts ranking |

结果解读：

如果老师问为什么 reranker 没有显著提升，可以回答：Hybrid 本身已经很强，而且小模型 reranker 在这个数据集上会改变前排排序。消融结果显示 Qwen-only 虽然让 doc_hit@10 达到 0.983，但 MRR 降到 0.825，所以最终选择保守融合。这不是失败，而是通过实验发现了当前资源条件下更稳的配置。

## 6. 生成与证据引用

在 RAG 回答阶段，系统会把检索到的 top chunks 组织成带编号的 context。每个 context block 包含标题、适用年份、来源 URL 和正文内容。

如果启用本地 LLM，系统会把这些官方资料和用户问题一起发送给本地 `Qwen2.5-0.5B-Instruct`。系统 prompt 要求模型只能依据提供的官方资料回答，关键事实必须使用 `[1]`、`[2]` 这样的编号引用证据，不得编造 URL、电话、日期、课程代码、学分、政策或来源。

系统 prompt 的核心约束是：

```text
你是南方科技大学校园知识助手。你只能依据提供的官方资料回答。
关键事实必须使用 [1]、[2] 这样的编号引用证据。
如果资料不足、证据冲突或问题超出资料范围，明确说明无法根据当前资料回答。
不得编造 URL、电话、日期、课程代码、学分、政策或来源。
```

如果不启用本地 LLM，系统会使用 extractive answer，也就是从检索证据中抽取最相关片段组织回答。现场演示默认推荐这种方式，因为它对学分、邮箱、电话、开放时间这类精确事实更稳定，不容易被小模型改写错。

所以本项目不是只依赖 prompt 控制幻觉，而是把控制分成三层：

第一层是检索，只把官方资料放入上下文。

第二层是 evidence gate，在生成前判断证据是否足够。

第三层是系统 prompt 或抽取式回答，限制最终回答必须基于证据。

## 7. Evidence Gate 证据充足性判断

Evidence gate 是本项目中控制幻觉风险的关键模块。它的作用是在生成之前判断：检索到的内容是否足以支持用户问题。

它不是单纯根据一个分数阈值决定，也不是让 LLM 自己判断。当前实现主要是规则型证据检查，结合了年份、限定词、重要词覆盖和少量 score fallback。

具体来说，系统会先把 top 5 检索结果的标题和正文拼成 combined evidence，然后做几类判断。

第一，如果没有检索结果，直接判定为 `insufficient_evidence`。

第二，如果问题里出现明确年份，比如 2024，而检索结果中没有对应年份或 metadata 中没有匹配年份，也判定证据不足。

第三，如果问题里出现具体校区限定，比如某个不存在的校区，系统会把它作为 required qualifier。如果这个限定词没有出现在检索证据中，也判定证据不足。

第四，系统会从问题中抽取重要词，过滤掉“南方科技大学”“什么”“相关”“资料”“怎么”等泛化词，然后检查这些重要词在证据中的覆盖率。如果多个具体重要词都没有被证据覆盖，就认为证据不足。

第五，如果重要词覆盖率比较高，则判定为 `supported`；如果覆盖率中等但最高检索分数较高，可以判定为 `partially_supported`；否则判定为 `insufficient_evidence`。

这个模块的意义是：检索到一些相关网页，并不等于可以回答用户问题。比如用户问“量子传送门预约服务”，系统可能检索到预约、服务、校园相关页面，但这些页面并不支持“量子传送门”这个具体事实，所以必须拒答。

Evidence gate 消融：

| setting | citation correct | false refusal | refusal accuracy | unanswerable refusal |
| --- | ---: | ---: | ---: | ---: |
| evidence gate enabled | 0.917 | 0.000 | 1.000 | 1.000 |
| force answer without gate | 0.917 | 0.000 | 0.923 | 0.000 |

结果解读：

这张表说明，如果关闭 evidence gate，只要检索到内容就强制回答，不可回答问题的拒答率会从 1.000 下降到 0.000。也就是说，系统会失去对不可回答问题的防护能力。因此 evidence gate 是这个系统可信性的关键，不是可有可无的后处理。

## 8. No-RAG 与 RAG 对比

为了说明 RAG 的必要性，系统保留了 `no_rag` 模式。在 No-RAG 模式下，本地 LLM 只接收用户问题，不进行检索，也没有官方证据上下文。

No-RAG 的问题是，它只能依赖模型参数知识。对于学校官网、培养方案 PDF、服务电话、开放时间这类内容，模型可能不知道最新信息，也可能回答得很自然但没有来源。

RAG 模式的区别在于：回答前先检索官方证据，回答后返回 citation。如果证据不足，系统拒答。因此 RAG 带来的不只是准确率提升，更重要的是可追踪性和安全性。

对比示例：

| question | No-RAG behavior | RAG behavior |
| --- | --- | --- |
| `2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？` | 可能不知道该 PDF，或者根据常识猜测。 | 检索官方 PDF，返回 `159学分` 并附 citation。 |
| `南方科技大学是否提供量子传送门预约服务？` | 可能用对话方式尝试回答。 | 返回 `insufficient_evidence`，不编造不存在服务。 |

## 9. 实验设计

实验分为两类：检索实验和生成/拒答实验。

检索实验主要评估系统能不能把正确文档或正确 chunk 排到前面。使用的指标包括：

- `doc_hit@5`：前 5 个结果中是否命中正确文档。
- `doc_hit@10`：前 10 个结果中是否命中正确文档。
- `chunk_hit@5`：前 5 个结果中是否命中正确 chunk。
- `chunk_hit@10`：前 10 个结果中是否命中正确 chunk。
- `MRR@10`：正确结果排名越靠前，分数越高。

生成和拒答实验主要评估最终回答是否可靠。使用的指标包括：

- `citation_correct_rate`：回答引用是否命中正确来源。
- `false_refusal_rate`：本来可以回答的问题是否被错误拒答。
- `refusal_accuracy`：拒答判断总体是否正确。
- `unanswerable_refusal_rate`：不可回答问题是否成功拒答。

我还额外做了消融实验，包括检索模式消融、reranker 权重消融、generation mode 消融和 evidence gate 消融。

## 10. 检索实验结果

下面是 test set 上的检索模式对比。

| mode | doc_hit@5 | doc_hit@10 | chunk_hit@5 | MRR@10 | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.933 | 0.967 | 0.883 | 0.871 | best for exact facts |
| Dense | 0.983 | 0.983 | 0.800 | 0.873 | best document recall |
| Hybrid | 0.967 | 0.967 | 0.817 | 0.877 | best overall ranking |
| Hybrid + Rerank | 0.917 | 0.967 | 0.817 | 0.876 | conservative reranking |

从这张表可以看到，BM25 的 chunk_hit@5 是最高的，达到 0.883。这说明对于精确事实，比如学分、电话、邮箱、开放时间，关键词检索非常有效。

Dense 的 doc_hit@5 和 doc_hit@10 都是 0.983，是文档级召回最强的。这说明语义检索更容易找到正确文档，即使用户表达和原文不完全一致。

Hybrid 的 MRR@10 是 0.877，是这组实验中最高的。这说明 BM25 和 Dense 是互补的，融合之后整体排序更稳。

Hybrid+Rerank 的 doc_hit@10 是 0.967，MRR@10 是 0.876，和 Hybrid 非常接近，但 doc_hit@5 略低。这也说明当前小 reranker 在 test 集上没有稳定提升前 5 位排序，所以最终报告中如实保留这个 tradeoff。

结果解读：

如果老师问“最终到底哪个最好”，可以回答：从 MRR 看 Hybrid 最好；从系统完整性看，我保留 Hybrid+Rerank 作为推荐展示模式，因为它体现了完整 RAG pipeline，同时通过消融说明 reranker 在当前资源和数据条件下是保守提升，不夸大结果。

## 11. 生成与拒答实验结果

在生成和拒答评估中，我使用 `hybrid_rerank` 作为检索模式，对 test set 进行评估。

核心结果是：

| metric | value |
| --- | ---: |
| cases | 65 |
| citation_correct_rate | 0.917 |
| false_refusal_rate | 0.000 |
| refusal_accuracy | 1.000 |
| unanswerable_refusal_rate | 1.000 |

这组结果说明三点。

第一，citation correct rate 是 0.917，说明大多数回答能引用到正确来源。

第二，false refusal rate 是 0，说明系统没有把本来可以回答的问题错误拒答。

第三，unanswerable refusal rate 是 1.0，说明不可回答问题全部被拒答。这一点对于校园信息问答很重要，因为系统不能为了显得“聪明”而编造不存在的政策或服务。

生成模式消融：

| generation mode | citation correct | false refusal | refusal accuracy | unanswerable refusal |
| --- | ---: | ---: | ---: | ---: |
| evidence-extractive | 0.917 | 0.000 | 1.000 | 1.000 |
| local LLM | 0.917 | 0.000 | 1.000 | 1.000 |

这张表说明，抽取式回答和本地 LLM 生成共享同一个 evidence gate，所以拒答指标保持一致。现场演示默认使用抽取式回答，是为了保证精确事实稳定；如果要展示自然语言生成能力，可以勾选 `Use local LLM`。

## 12. 现场演示流程

下面进入系统演示。演示时我会打开 Gradio 页面，默认设置为：

```text
Retrieval mode: Hybrid + Rerank
Use local LLM: unchecked
```

这里先不勾选本地 LLM，是为了优先展示最稳定的证据抽取能力。对于学分、电话、邮箱、开放时间这类精确事实，抽取式回答更容易保持原文表达。主线演示只使用已经通过 API 验证的三个问题：网页信息、PDF 信息和不可回答问题拒答。

主线问题已在当前服务上验证：

| demo | question | expected status | stable evidence |
| --- | --- | --- | --- |
| 图书馆开放时间 | `南方科技大学图书馆开放时间是什么？` | `supported` | 图书馆页面，包含 `8:00 - 22:00` |
| PDF 培养方案 | `2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？` | `supported` | 机器人工程培养方案 PDF，包含 `159学分` |
| 拒答能力 | `南方科技大学是否提供量子传送门预约服务？` | `insufficient_evidence` | 无足够官方证据，系统拒答 |

### Demo 1: 图书馆开放时间

问题：

```text
南方科技大学图书馆开放时间是什么？
```

预期展示：

- Evidence status 是 `supported`。
- Sources 包含图书馆官方页面。
- 回答中应出现 `周一至周日：8:00 - 22:00`。
- 如果检索到对应证据，也会显示 `一丹一楼24小时开放`。

这个问题展示的是网页类信息检索。系统先从图书馆相关网页中检索证据，再基于证据回答。这里可以看到回答不是模型凭空说的，而是下面有 source cards 和 evidence snippets。

### Demo 2: PDF 培养方案

问题：

```text
2024级机器人工程专业本科人才培养方案的最低毕业学分要求是多少？
```

预期答案：

```text
159 学分
```

预期展示：

- Citation 标题包含 `12-2024级机器人工程专业本科人才培养方案`。
- 这说明系统成功处理了 PDF，而不是只处理网页。

这个问题展示 PDF 检索能力。项目中的 PDF 不是手动复制答案，而是通过脚本从公开来源下载，解析后和网页一样进入 chunk 和索引。这里系统找到机器人工程专业的培养方案，并从 PDF 中抽取最低毕业学分要求。

如果老师想看生成能力，可以在这个问题上勾选 `Use local LLM`。已经验证本地 Qwen2.5-0.5B 会基于证据回答 `159学分`。

### Demo 3: 不可回答问题拒答

问题：

```text
南方科技大学是否提供量子传送门预约服务？
```

预期答案：

```text
当前知识库没有检索到足够直接证据，无法根据现有官方资料回答该问题。
```

这个问题用于展示拒答机制。注意这里并不是简单地“没搜到东西”，而是检索结果不能支持问题中的具体事实。系统判断证据不足后，不会把校园服务页面强行拼成一个答案，而是返回 `insufficient_evidence`。

### 可选补充: 心理咨询预约

如果现场时间充足，可以补充展示学生服务类问题：

```text
南方科技大学心理咨询怎么预约？
```

在 `Hybrid + Rerank` 和不使用 LLM 的设置下，该问题已通过 API 验证，能够返回邮箱预约 `counseling@sustc.edu.cn`、电话预约 `88010576` 和地点 `荔园9栋一层105`。这个问题不建议单独切到 `Dense` 模式演示，因为 Dense 在该问题上可能召回教学工作部联系方式，现场观感不如主线三问稳定。

### 可选补充: 检索模式切换

如果时间允许，我可以切换 BM25、Dense、Hybrid 和 Hybrid+Rerank。BM25 更适合关键词精确匹配，Dense 更适合语义召回，Hybrid 通过 RRF 融合两者，Hybrid+Rerank 再做相关性重排序。前端 trace 区域可以看到不同阶段的检索结果。

## 13. 系统服务与部署

系统运行时有三个服务。

第一是本地 LLM 服务，端口是 8000，负责提供 OpenAI-compatible chat completion 接口。

第二是 RAG API 服务，端口是 8080，负责接收问题、调用检索、执行 evidence gate 和生成回答。

第三是 Gradio UI，端口是 7860，用于现场演示。

服务表：

| service | port | command |
| --- | ---: | --- |
| local LLM | 8000 | `bash scripts/run_llm.sh` |
| RAG API | 8080 | `bash scripts/run_api.sh` |
| Gradio UI | 7860 | `bash scripts/run_ui.sh` |

一个实现上的细节是，Gradio 前端不会直接读取 Qdrant 索引，而是统一调用 API。因为当前使用 embedded Qdrant，本地文件会有锁。如果前端和 API 同时直接打开向量库，可能出现文件锁冲突。通过 API 统一访问后，Qdrant 只由 API 进程管理，演示更稳定。

健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8080/health
```

## 14. 可复现性

项目的可复现性体现在三个方面。

第一，数据源写在配置文件中，不依赖人工临时复制。

第二，处理流水线由脚本串起来，可以从 crawl、PDF fetch、parse、chunk、index 一路重建。

第三，评估和消融脚本固定输出结果文件，实验报告可以重新生成。

完整重建可以按以下顺序运行：

```bash
cd /root/RAG_proj/sustech-rag
bash scripts/run_pipeline_core.sh
python scripts/fetch_pdf_sources.py
python scripts/parse.py
python scripts/chunk.py
python scripts/build_bm25.py
python scripts/embed.py
python scripts/generate_eval.py
python scripts/evaluate.py --split dev
python scripts/evaluate.py --split test
python scripts/evaluate_generation.py --split test --mode hybrid_rerank --use-llm
python scripts/run_ablations.py --split test
python scripts/report_summary.py
```

需要注意的是，embedded Qdrant 有本地文件锁，所以直接重建索引和评估时最好顺序运行。如果 API 正在占用 Qdrant，直接写索引前应先停掉 API，或者通过 API 做查询类评估。

## 15. 项目贡献总结

我认为这个项目的主要贡献有四点。

第一，构建了一个覆盖多个南科大公开官方来源的校园知识库，包括网页和 PDF，两类数据都进入统一的 RAG pipeline。

第二，实现了多种检索架构，包括 BM25、Dense、Hybrid 和 Hybrid+Rerank，并用实验比较了它们在 document hit、chunk hit 和 MRR 上的差异。

第三，实现了证据充足性判断和拒答机制。系统不是只要检索到相关内容就回答，而是会检查年份、限定词和重要词覆盖，证据不足时拒答。

第四，完成了本地可运行的 API 和 Gradio demo，并提供了实验报告、消融报告、演示脚本和操作文档，使整个项目可以展示、复现和解释。

一句话总结：

这个项目的重点不是让模型自由发挥，而是把公开官方资料组织成可检索知识库，通过 RAG 让回答具备来源、证据和拒答能力。

## 16. 常见追问准备

### Q1: 为什么不用纯 LLM 直接回答？

纯 LLM 直接回答依赖参数知识，可能不知道最新校园信息，也可能编造。比如 2024 级具体专业培养方案、图书馆开放时间、心理咨询电话，这些都需要官方资料支持。RAG 的好处是回答前先检索官方证据，回答后可以展示 citation。如果证据不足，系统会拒答。

### Q2: BM25 和 Dense 为什么都要做？

BM25 对精确词更强，比如年份、专业名、学分、电话、邮箱。Dense 对语义相近但措辞不同的问题更强。校园问答同时存在这两类需求，所以只用一种检索不够稳。实验也显示，BM25 的 chunk hit 更高，Dense 的 document recall 更强，Hybrid 的 MRR 最好。

### Q3: Hybrid 是怎么融合的？

使用 RRF，也就是 Reciprocal Rank Fusion。它不直接比较 BM25 和 Dense 的原始分数，而是看每个 chunk 在不同检索结果中的排名。排名越靠前，贡献越大。这样可以避免不同检索器分数尺度不一致的问题。

### Q4: Reranker 为什么没有明显提升？

当前 reranker 是 0.6B 小模型，而 Hybrid 本身已经很强。消融显示，越依赖 Qwen reranker，chunk hit 略有提升，但 MRR 下降。Qwen-only 的 doc_hit@10 能到 0.983，但 MRR 降到 0.825，说明它能找到相关文档，却不一定把最佳证据排到最前。所以我采用了保守融合，并如实报告 tradeoff。

### Q5: Evidence gate 是谁判断的？

不是 LLM 自己判断，也不是单纯 score 阈值。当前是代码中的规则型证据检查，检查 top 5 检索结果是否覆盖问题中的年份、具体限定词和重要词。如果证据覆盖不足，就返回 `insufficient_evidence`。只有在覆盖中等且最高检索分数较高时，才会给 `partially_supported`。

### Q6: Evidence gate 为什么重要？

因为检索到相关网页不等于证据足够。比如问题里出现不存在的“量子传送门预约服务”，系统可能检索到预约或服务相关页面，但这些页面不能支持该事实。消融结果显示，关闭 evidence gate 后，不可回答问题拒答率从 1.0 下降到 0，所以它是降低幻觉风险的关键模块。

### Q7: 抽取式回答合理吗？

合理。抽取式回答不是替代 RAG，而是 RAG 生成阶段的一种安全模式。对于学分、电话、邮箱、开放时间这类精确事实，直接从证据中抽取比让 0.5B 小模型改写更稳定。项目同时保留 local LLM 模式，用于展示自然语言生成。两者共用同一套检索和 evidence gate。

### Q8: 为什么使用 0.5B 生成模型？

主要考虑 AutoDL 环境和显存稳定性。这个项目的重点是 RAG pipeline，而不是追求最大生成模型。0.5B 模型通过本地 OpenAI-compatible endpoint 接入，足以展示基于证据的生成流程。对于精确事实，演示默认用抽取式回答保证稳定。如果资源允许，可以换成更大的 Qwen2.5-7B-Instruct。

### Q9: PDF 是怎么进入知识库的？

PDF 来源写在配置中，`fetch_pdf_sources.py` 会从公开目录发现并下载 PDF。`parse.py` 使用 PyMuPDF 提取文本和页码，之后 PDF 与 HTML 一样进入 normalized documents、chunk、BM25 和 Qdrant 向量索引。这样 PDF 问题可以正常检索并返回 citation。

### Q10: GitHub 为什么没有大模型和全部数据？

因为模型、raw data 和索引属于大体积运行时 artifacts，不适合直接放入 GitHub。它们保存在 AutoDL 挂载目录 `/root/autodl-fs/sustech-rag`。GitHub 仓库保留代码、配置、脚本和文档，足够说明结构并支持重新构建。

### Q11: 如果现场 LLM 服务不稳定怎么办？

演示默认不依赖 LLM 生成，而是使用 extractive answer。只要 API 和检索服务正常，就能展示核心 RAG 能力。如果需要展示 LLM，可以只在 PDF 学分问题上勾选 `Use local LLM`，该问题已经验证会回答 `159学分`。

### Q12: 后续可以怎么改进？

后续可以从三个方向改进。第一，扩大官方数据覆盖范围，加入更多部门页面。第二，改进 chunk 策略，例如加入 parent-child chunk retrieval，让系统先用小 chunk 精确召回，再把父级上下文送入生成。第三，使用更大的 reranker 或人工标注数据做 reranker calibration，提高前排排序稳定性。

## 17. 最后收尾

总结一下，我的项目实现了一个完整的南科大校园 RAG 问答系统。它从公开官方网页和 PDF 自动构建知识库，支持 BM25、Dense、Hybrid 和 Hybrid+Rerank 多种检索策略，使用本地模型完成生成，并通过 evidence gate 控制证据不足时的拒答。

实验结果表明，Hybrid 检索在 MRR 上表现最好，Dense 在文档召回上表现最好，BM25 在精确 chunk 命中上表现突出；生成评估中 citation correct rate 达到 0.917，不可回答问题拒答率达到 1.000。

因此，这个项目不仅实现了一个能运行的问答 demo，也完成了数据、检索、生成、安全拒答和实验消融的完整闭环。
