from __future__ import annotations

from threading import Lock
from pathlib import Path

from sustech_rag.common.config import PROJECT_ROOT, load_yaml
from sustech_rag.common.schema import RetrievalHit
from sustech_rag.indexing.bm25 import BM25Retriever
from sustech_rag.reranking.reranker import get_reranker
from sustech_rag.retrieval.dense import DenseRetriever
from sustech_rag.retrieval.fusion import reciprocal_rank_fusion
from sustech_rag.retrieval.query import analyze_query


class RetrievalService:
    def __init__(self, config_path: str | Path = PROJECT_ROOT / "configs" / "retrieval.yaml") -> None:
        self.config = load_yaml(config_path)
        self._bm25: BM25Retriever | None = None
        self._dense: DenseRetriever | None = None
        self._init_lock = Lock()

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            with self._init_lock:
                if self._bm25 is None:
                    self._bm25 = BM25Retriever()
        return self._bm25

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            with self._init_lock:
                if self._dense is None:
                    self._dense = DenseRetriever()
        return self._dense

    def retrieve(self, query: str, mode: str = "bm25", context_top_k: int | None = None) -> dict:
        analysis = analyze_query(query)
        trace = {"analysis": analysis.__dict__, "mode": mode}
        bm25_hits: list[RetrievalHit] = []
        dense_hits: list[RetrievalHit] = []
        if mode in {"bm25", "hybrid", "hybrid_rerank"}:
            bm25_hits = self.bm25.search(query, top_k=int(self.config.get("bm25_top_k", 30)))
            trace["bm25"] = [h.model_dump(mode="json") for h in bm25_hits]
        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            dense_hits = self.dense.search(query, top_k=int(self.config.get("dense_top_k", 30)))
            trace["dense"] = [h.model_dump(mode="json") for h in dense_hits]
        if mode == "dense":
            hits = dense_hits
        elif mode in {"hybrid", "hybrid_rerank"}:
            hits = reciprocal_rank_fusion(
                [hit_set for hit_set in (dense_hits, bm25_hits) if hit_set],
                k=int(self.config.get("rrf_k", 60)),
                top_k=int(self.config.get("fusion_top_k", 30)),
            )
            trace["rrf"] = [h.model_dump(mode="json") for h in hits]
            if mode == "hybrid_rerank":
                hits = get_reranker().rerank(
                    query,
                    hits[: int(self.config.get("rerank_top_k", 30))],
                    top_k=int(self.config.get("fusion_top_k", 30)),
                )
                trace["rerank"] = [h.model_dump(mode="json") for h in hits]
        else:
            hits = bm25_hits
        hits = self._deduplicate(hits)
        top_k = int(context_top_k or self.config.get("context_top_k", 5))
        return {"hits": hits[:top_k], "trace": trace}

    def _deduplicate(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        per_doc_limit = int(self.config.get("per_doc_limit", 3))
        doc_counts: dict[str, int] = {}
        text_hashes: set[str] = set()
        kept: list[RetrievalHit] = []
        for hit in hits:
            snippet_key = hit.text[:180]
            if snippet_key in text_hashes:
                continue
            if doc_counts.get(hit.doc_id, 0) >= per_doc_limit:
                continue
            text_hashes.add(snippet_key)
            doc_counts[hit.doc_id] = doc_counts.get(hit.doc_id, 0) + 1
            kept.append(hit)
        for rank, hit in enumerate(kept, start=1):
            hit.rank = rank
        return kept
