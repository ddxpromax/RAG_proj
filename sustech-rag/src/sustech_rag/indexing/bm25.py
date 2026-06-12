from __future__ import annotations

import pickle
import re
from pathlib import Path

from sustech_rag.chunking.chunker import load_chunks
from sustech_rag.common.config import load_paths
from sustech_rag.common.logging import get_logger
from sustech_rag.common.schema import Chunk, RetrievalHit

logger = get_logger(__name__)


TOKEN_RE = re.compile(r"[A-Za-z]{1,8}\d{1,5}|20[0-3]\d|[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    try:
        import jieba

        words = list(jieba.cut(text))
        code_words = TOKEN_RE.findall(text)
        return [w.strip().lower() for w in words + code_words if w.strip()]
    except Exception:
        return [w.lower() for w in TOKEN_RE.findall(text)]


def build_bm25(chunks: list[Chunk] | None = None) -> Path:
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("rank-bm25 is required. Install `.[full]`.") from exc

    chunks = chunks or load_chunks()
    corpus = [tokenize(chunk.embedding_text) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    payload = {"bm25": bm25, "chunks": [chunk.model_dump() for chunk in chunks]}
    paths = load_paths()
    out = Path(paths["indexes"]["bm25"]) / "bm25.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(payload, f)
    logger.info("Wrote BM25 index with %s chunks to %s", len(chunks), out)
    return out


class BM25Retriever:
    def __init__(self, index_path: str | Path | None = None) -> None:
        paths = load_paths()
        self.index_path = Path(index_path or Path(paths["indexes"]["bm25"]) / "bm25.pkl")
        with self.index_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.chunks = [Chunk(**row) for row in payload["chunks"]]

    def search(self, query: str, top_k: int = 30) -> list[RetrievalHit]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        hits: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.display_text,
                    title=chunk.metadata.get("title"),
                    url=chunk.metadata.get("url"),
                    score=float(score),
                    rank=rank,
                    source="bm25",
                    metadata=chunk.metadata,
                )
            )
        return hits

