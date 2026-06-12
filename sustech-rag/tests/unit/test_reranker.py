from sustech_rag.common.schema import RetrievalHit
from sustech_rag.reranking.reranker import Reranker


def test_lexical_reranker_prefers_overlap(tmp_path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text("reranker:\n  backend: lexical\n", encoding="utf-8")
    hits = [
        RetrievalHit(chunk_id="1", doc_id="d1", text="无关内容", score=10, rank=1, source="hybrid"),
        RetrievalHit(chunk_id="2", doc_id="d2", text="图书馆 开馆时间 空间预约", score=1, rank=2, source="hybrid"),
    ]
    ranked = Reranker(config_path=config_path).rerank("图书馆开放时间", hits, top_k=2)
    assert ranked[0].chunk_id == "2"
    assert ranked[0].source.endswith("+rerank")
