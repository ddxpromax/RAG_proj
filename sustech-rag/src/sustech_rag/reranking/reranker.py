from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sustech_rag.common.config import PROJECT_ROOT, load_yaml
from sustech_rag.common.schema import RetrievalHit


TOKEN_RE = re.compile(r"[A-Za-z]{1,8}\d{0,5}|20[0-3]\d|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [token.lower() for token in TOKEN_RE.findall(lowered)]
    try:
        import jieba

        tokens.extend(word.strip().lower() for word in jieba.cut(lowered) if len(word.strip()) > 1)
    except Exception:
        pass
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", lowered)
    tokens.extend(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
    return tokens


class Reranker:
    def __init__(self, config_path: str | Path = PROJECT_ROOT / "configs" / "models.yaml") -> None:
        self.config = load_yaml(config_path)["reranker"]
        self.backend = self.config.get("backend", "lexical")
        self.batch_size = int(self.config.get("batch_size", 8))
        self.max_length = int(self.config.get("max_length", 2048))
        self.qwen_score_weight = float(self.config.get("qwen_score_weight", 0.25))
        self.instruction = self.config.get(
            "instruction",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        self.model = None
        self.tokenizer = None
        self._true_token_id = None
        self._false_token_id = None
        self._prefix_tokens = None
        self._suffix_tokens = None
        if self.backend == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for cross-encoder reranking.") from exc
            model_id = self.config["local_path"] if Path(self.config["local_path"]).exists() else self.config["model_id"]
            self.model = CrossEncoder(str(model_id), trust_remote_code=True)
        elif self.backend == "qwen_causal":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("transformers and torch are required for Qwen causal reranking.") from exc
            model_id = self.config["local_path"] if Path(self.config["local_path"]).exists() else self.config["model_id"]
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_id), padding_side="left", trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_id),
                dtype=dtype,
                trust_remote_code=True,
            ).eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self._false_token_id = self.tokenizer.convert_tokens_to_ids("no")
            self._true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            prefix = (
                "<|im_start|>system\n"
                'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                'Note that the answer can only be "yes" or "no".<|im_end|>\n'
                "<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 30) -> list[RetrievalHit]:
        if not hits:
            return []
        if self.backend == "cross_encoder":
            return self._cross_encoder_rerank(query, hits, top_k)
        if self.backend == "qwen_causal":
            return self._qwen_causal_rerank(query, hits, top_k)
        return self._lexical_rerank(query, hits, top_k)

    def _cross_encoder_rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        assert self.model is not None
        scores = self.model.predict([(query, hit.text) for hit in hits], batch_size=self.batch_size)
        reranked = []
        for hit, score in zip(hits, scores, strict=True):
            copied = hit.model_copy(deep=True)
            copied.metadata["pre_rerank_score"] = copied.score
            copied.metadata["pre_rerank_rank"] = copied.rank
            copied.score = float(score)
            copied.source = f"{copied.source}+rerank"
            reranked.append(copied)
        return self._finish(reranked, top_k)

    def _qwen_causal_rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        assert self.model is not None
        assert self.tokenizer is not None
        assert self._prefix_tokens is not None
        assert self._suffix_tokens is not None
        assert self._true_token_id is not None
        assert self._false_token_id is not None
        import torch

        scores: list[float] = []
        pairs = [self._format_instruction(query, hit.text) for hit in hits]
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
            )
            for index, token_ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][index] = self._prefix_tokens + token_ids + self._suffix_tokens
            encoded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits[:, -1, :]
                binary_logits = torch.stack(
                    [logits[:, self._false_token_id], logits[:, self._true_token_id]],
                    dim=1,
                )
                batch_scores = torch.nn.functional.log_softmax(binary_logits, dim=1)[:, 1].exp()
            scores.extend(float(score) for score in batch_scores.detach().cpu())
        reranked = []
        total_hits = max(1, len(hits))
        qwen_weight = min(1.0, max(0.0, self.qwen_score_weight))
        for hit, score in zip(hits, scores, strict=True):
            copied = hit.model_copy(deep=True)
            copied.metadata["pre_rerank_score"] = copied.score
            copied.metadata["pre_rerank_rank"] = copied.rank
            rank_prior = (total_hits - copied.rank + 1) / total_hits
            copied.metadata["qwen_rerank_score"] = score
            copied.metadata["rank_prior_score"] = rank_prior
            copied.score = qwen_weight * score + (1.0 - qwen_weight) * rank_prior
            copied.source = f"{copied.source}+rerank"
            reranked.append(copied)
        return self._finish(reranked, top_k)

    def _format_instruction(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _lexical_rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        query_terms = Counter(tokenize(query))
        reranked: list[RetrievalHit] = []
        for hit in hits:
            doc_terms = Counter(tokenize(hit.text))
            overlap = 0.0
            for term, q_count in query_terms.items():
                if term in doc_terms:
                    overlap += min(q_count, doc_terms[term]) * (2.0 if len(term) >= 2 else 0.0)
            title_bonus = 0.0
            title = (hit.title or "").lower()
            for term in query_terms:
                if len(term) > 1 and term in title:
                    title_bonus += 2.5
            category_bonus = 0.0
            category = str(hit.metadata.get("category") or "")
            if "图书馆" in query and category == "library":
                category_bonus += 3.0
            if any(word in query for word in ("培养", "专业", "学分", "课程")) and category == "undergraduate_teaching":
                category_bonus += 2.0
            if any(word in query for word in ("书院", "学生事务", "奖学金", "心理")) and category == "student_affairs":
                category_bonus += 2.0
            length_penalty = 1.0 / math.sqrt(max(1, len(hit.text) / 300))
            copied = hit.model_copy(deep=True)
            copied.metadata["pre_rerank_score"] = copied.score
            copied.metadata["pre_rerank_rank"] = copied.rank
            copied.score = float(overlap * length_penalty + title_bonus + category_bonus + 0.05 * hit.score)
            copied.source = f"{copied.source}+rerank"
            reranked.append(copied)
        return self._finish(reranked, top_k)

    @staticmethod
    def _finish(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        ranked = sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
        for rank, hit in enumerate(ranked, start=1):
            hit.rank = rank
        return ranked


@lru_cache(maxsize=2)
def get_reranker() -> Reranker:
    return Reranker()
