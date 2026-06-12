from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import hashlib
import re

import numpy as np

from sustech_rag.common.config import PROJECT_ROOT, configure_model_cache, load_paths, load_yaml


class EmbeddingModel:
    def __init__(self, config_path: str | Path = PROJECT_ROOT / "configs" / "models.yaml") -> None:
        configure_model_cache()
        config = load_yaml(config_path)["embedding"]
        self.backend = config.get("backend", "sentence_transformers")
        self.model_id = config.get("local_path") if Path(config.get("local_path", "")).exists() else config["model_id"]
        self.batch_size = int(config.get("batch_size", 16))
        self.normalize = bool(config.get("normalize", True))
        self.hashing_dimension = int(config.get("hashing_dimension", 768))
        self.max_length = int(config.get("max_length", 1024))
        self.query_prompt_name = config.get("query_prompt_name")
        self.model = None
        if self.backend == "hashing":
            return
        paths = load_paths()
        cache_folder = paths["models"]["hf_home"]
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for dense embeddings.") from exc
        self.model = SentenceTransformer(str(self.model_id), cache_folder=cache_folder, trust_remote_code=True)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = self.max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "hashing":
            return self._hashing_encode(texts)
        assert self.model is not None
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.backend == "hashing" or not self.query_prompt_name:
            return self.encode(texts)
        assert self.model is not None
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name=self.query_prompt_name,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _hashing_encode(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.hashing_dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = self._tokens(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.hashing_dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                rows[row_index, bucket] += sign
            norm = float(np.linalg.norm(rows[row_index]))
            if self.normalize and norm > 0:
                rows[row_index] /= norm
        return rows

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z]{1,8}\d{0,5}|20[0-3]\d|\d+(?:\.\d+)?|[\u4e00-\u9fff]", lowered)
        char_bigrams = [lowered[i : i + 2] for i in range(max(0, len(lowered) - 1)) if "\n" not in lowered[i : i + 2]]
        return words + char_bigrams


@lru_cache(maxsize=2)
def get_embedding_model(config_path: str = str(PROJECT_ROOT / "configs" / "models.yaml")) -> EmbeddingModel:
    return EmbeddingModel(Path(config_path))
