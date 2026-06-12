from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from sustech_rag.common.config import PROJECT_ROOT, load_paths
from sustech_rag.common.logging import get_logger
from sustech_rag.common.schema import Chunk, Document
from sustech_rag.common.utils import sha256_text
from sustech_rag.parsing.pipeline import load_documents

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def split_sections(text: str) -> list[tuple[list[str], str, int | None, int | None]]:
    sections: list[tuple[list[str], str, int | None, int | None]] = []
    current_headings: list[str] = []
    current_lines: list[str] = []
    current_page: int | None = None
    for line in text.splitlines():
        page_match = re.match(r"\[PDF_PAGE\s+(\d+)\]", line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if re.match(r"^\s*(第[一二三四五六七八九十0-9]+[章节条]|[一二三四五六七八九十0-9]+[、.])", line):
            if current_lines:
                sections.append((current_headings[:], "\n".join(current_lines).strip(), current_page, current_page))
                current_lines = []
            current_headings = [line.strip()]
        current_lines.append(line)
    if current_lines:
        sections.append((current_headings[:], "\n".join(current_lines).strip(), current_page, current_page))
    return [(h, body, ps, pe) for h, body, ps, pe in sections if body.strip()]


def window_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = text.rfind("\n", start, end)
        if cut <= start + max_chars // 2:
            cut = end
        windows.append(text[start:cut].strip())
        if cut >= len(text):
            break
        start = max(0, cut - overlap)
    return [w for w in windows if w]


def chunk_document(doc: Document, max_chars: int = 600, overlap: int = 70) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = split_sections(doc.text)
    for section_path, body, page_start, page_end in sections:
        for piece in window_text(body, max_chars=max_chars, overlap=overlap):
            idx = len(chunks)
            chunk_hash = sha256_text(f"{doc.doc_id}:{idx}:{piece}")
            heading = " / ".join(section_path)
            embedding_text = (
                f"文档：{doc.title}\n"
                f"适用年份：{doc.effective_year or '未知'}\n"
                f"章节：{heading or '正文'}\n"
                f"正文：{piece}"
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:chunk:{idx:04d}",
                    doc_id=doc.doc_id,
                    parent_id=doc.doc_id,
                    chunk_index=idx,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    text=piece,
                    embedding_text=embedding_text,
                    display_text=piece,
                    token_count=estimate_tokens(embedding_text),
                    hash=chunk_hash,
                    metadata={
                        "title": doc.title,
                        "url": doc.source_url,
                        "source_type": doc.source_type,
                        "category": doc.category,
                        "effective_year": doc.effective_year,
                        "status": doc.status,
                    },
                )
            )
    return chunks


def build_chunks(docs: list[Document] | None = None) -> list[Chunk]:
    paths = load_paths()
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "chunking.yaml").read_text(encoding="utf-8"))
    default = config["default"]
    docs = docs or load_documents()
    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    for doc in docs:
        doc_chunks = chunk_document(
            doc,
            max_chars=int(default["target_chars_max"]),
            overlap=int(default["overlap_chars"]),
        )
        for chunk in doc_chunks:
            if chunk.hash in seen_hashes:
                continue
            seen_hashes.add(chunk.hash)
            chunks.append(chunk)
    out_dir = Path(paths["data"]["chunks"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    logger.info("Wrote %s chunks to %s", len(chunks), out_path)
    return chunks


def load_chunks() -> list[Chunk]:
    paths = load_paths()
    path = Path(paths["data"]["chunks"]) / "chunks.jsonl"
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks

