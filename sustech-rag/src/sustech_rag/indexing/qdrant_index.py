from __future__ import annotations

import hashlib
from pathlib import Path

from sustech_rag.chunking.chunker import load_chunks
from sustech_rag.common.config import PROJECT_ROOT, load_paths, load_yaml
from sustech_rag.common.logging import get_logger
from sustech_rag.indexing.embeddings import EmbeddingModel

logger = get_logger(__name__)


def point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16], 16)


def build_qdrant(config_path: str | Path = PROJECT_ROOT / "configs" / "retrieval.yaml") -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qdrant-client is required for dense indexing.") from exc

    chunks = load_chunks()
    config = load_yaml(config_path)
    paths = load_paths()
    client = QdrantClient(path=paths["indexes"]["qdrant_storage"])
    embedder = EmbeddingModel()
    vectors = embedder.encode([chunk.embedding_text for chunk in chunks])
    collection = config.get("collection", "sustech_chunks_v1")
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
    )
    points = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.display_text,
                    "embedding_text": chunk.embedding_text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            )
        )
    for start in range(0, len(points), 128):
        client.upsert(collection_name=collection, points=points[start : start + 128])
    for field in ["category", "effective_year", "status", "doc_id", "source_type"]:
        try:
            schema = PayloadSchemaType.INTEGER if field == "effective_year" else PayloadSchemaType.KEYWORD
            client.create_payload_index(collection, field_name=field, field_schema=schema)
        except Exception:
            logger.debug("Payload index may already exist: %s", field)
    logger.info("Wrote %s dense vectors to local Qdrant collection=%s", len(points), collection)


def build_qdrant_placeholder(config_path: str | Path = PROJECT_ROOT / "configs" / "retrieval.yaml") -> None:
    build_qdrant(config_path)
