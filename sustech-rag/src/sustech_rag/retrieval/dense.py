from __future__ import annotations

from pathlib import Path

from sustech_rag.common.config import PROJECT_ROOT, load_paths, load_yaml
from sustech_rag.common.schema import RetrievalHit
from sustech_rag.indexing.embeddings import get_embedding_model


class DenseRetriever:
    def __init__(self, config_path: str | Path = PROJECT_ROOT / "configs" / "retrieval.yaml") -> None:
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client is required for dense retrieval.") from exc
        self.config = load_yaml(config_path)
        paths = load_paths()
        self.client = QdrantClient(path=paths["indexes"]["qdrant_storage"])
        self.collection = self.config.get("collection", "sustech_chunks_v1")
        self.embedder = get_embedding_model()

    def search(self, query: str, top_k: int = 30) -> list[RetrievalHit]:
        query_vector = self.embedder.encode_queries([query])[0].tolist()
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        rows = result.points
        hits: list[RetrievalHit] = []
        for rank, row in enumerate(rows, start=1):
            payload = row.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=payload.get("chunk_id", str(row.id)),
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", ""),
                    title=payload.get("title"),
                    url=payload.get("url"),
                    score=float(row.score),
                    rank=rank,
                    source="dense",
                    metadata=payload,
                )
            )
        return hits
