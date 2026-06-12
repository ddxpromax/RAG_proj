from sustech_rag.common.schema import RetrievalHit
from sustech_rag.retrieval.fusion import reciprocal_rank_fusion


def hit(chunk_id: str, rank: int, source: str) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, doc_id="d", text="t", score=1, rank=rank, source=source)


def test_rrf_prefers_shared_hits() -> None:
    fused = reciprocal_rank_fusion(
        [[hit("a", 1, "dense"), hit("b", 2, "dense")], [hit("b", 1, "bm25"), hit("c", 2, "bm25")]],
        k=60,
        top_k=3,
    )
    assert fused[0].chunk_id == "b"
    assert fused[0].source == "bm25+dense"

