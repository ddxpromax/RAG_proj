from __future__ import annotations

from collections import defaultdict

from sustech_rag.common.schema import RetrievalHit


def reciprocal_rank_fusion(result_sets: list[list[RetrievalHit]], k: int = 60, top_k: int = 30) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalHit] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    for hits in result_sets:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] += 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
            sources[hit.chunk_id].append(hit.source)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    fused: list[RetrievalHit] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        hit = best[chunk_id].model_copy()
        hit.score = scores[chunk_id]
        hit.rank = rank
        hit.source = "+".join(sorted(set(sources[chunk_id])))
        fused.append(hit)
    return fused

