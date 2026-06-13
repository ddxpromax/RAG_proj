# Presentation Tables

This file contains compact tables that can be copied directly into slides.

## Table 1: Corpus and System Scale

| item | value |
| --- | ---: |
| normalized documents | 285 |
| HTML documents | 243 |
| PDF documents | 42 |
| chunks | 2649 |
| embedding model | Qwen3-Embedding-0.6B |
| reranker model | Qwen3-Reranker-0.6B |
| generator model | Qwen2.5-0.5B-Instruct |

Speaker note:

```text
数据来自南科大公开官方来源，包括网页和 2024 级本科培养方案 PDF。系统不是手工塞答案，而是从公开网页/PDF 抽取、切块、建索引后检索回答。
```

## Table 2: Retrieval Mode Ablation

| mode | doc_hit@5 | doc_hit@10 | chunk_hit@5 | MRR@10 | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.933 | 0.967 | 0.883 | 0.871 | best for exact facts |
| Dense | 0.983 | 0.983 | 0.800 | 0.873 | best document recall |
| Hybrid | 0.967 | 0.967 | 0.817 | 0.877 | best overall ranking |
| Hybrid + Rerank | 0.917 | 0.967 | 0.817 | 0.876 | conservative reranking |

Speaker note:

```text
BM25 对学分、邮箱、电话这类精确词更强；Dense 更容易找对文档；Hybrid 的 MRR 最好，说明词法和语义召回互补。Reranker 在当前小模型和小数据集上没有稳定提升前五位，因此报告如实保留 tradeoff。
```

## Table 3: Reranker Weight Ablation

| qwen_score_weight | doc_hit@5 | doc_hit@10 | chunk_hit@5 | MRR@10 | interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.967 | 0.967 | 0.817 | 0.877 | original hybrid rank only |
| 0.25 | 0.917 | 0.967 | 0.817 | 0.876 | current conservative blend |
| 0.50 | 0.917 | 0.967 | 0.833 | 0.875 | slightly better chunk hit |
| 0.75 | 0.917 | 0.967 | 0.833 | 0.874 | similar chunk hit, lower MRR |
| 1.00 | 0.900 | 0.983 | 0.833 | 0.825 | Qwen-only over-shifts ranking |

Speaker note:

```text
越相信 Qwen reranker，chunk hit 略升，但 MRR 下降。说明小 reranker 能识别相关片段，却不一定把最佳证据放最前，所以最终采用保守融合。
```

## Table 4: Refusal Gate Ablation

| setting | citation correct | false refusal | refusal accuracy | unanswerable refusal |
| --- | ---: | ---: | ---: | ---: |
| evidence gate enabled | 0.917 | 0.000 | 1.000 | 1.000 |
| force answer without gate | 0.917 | 0.000 | 0.923 | 0.000 |

Speaker note:

```text
如果不做证据充足性判断，只要检索到内容就强制回答，不可回答题的拒答率会从 1.0 掉到 0。这说明 refusal gate 是安全性的关键模块。
```

## Table 5: Generation Mode Ablation

| generation mode | citation correct | false refusal | refusal accuracy | unanswerable refusal |
| --- | ---: | ---: | ---: | ---: |
| evidence-extractive | 0.917 | 0.000 | 1.000 | 1.000 |
| local LLM | 0.917 | 0.000 | 1.000 | 1.000 |

Speaker note:

```text
两种生成方式共用同一个 evidence gate，所以拒答指标一致。现场演示默认使用 extractive，是为了保证电话、邮箱、学分这类精确事实不被小模型改写错；local LLM 可作为自然语言生成能力展示。
```
