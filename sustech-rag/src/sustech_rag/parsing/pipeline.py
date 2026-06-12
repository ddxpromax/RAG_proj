from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from sustech_rag.common.config import load_paths
from sustech_rag.common.logging import get_logger
from sustech_rag.common.schema import Document, RawManifest
from sustech_rag.common.utils import sha256_text
from sustech_rag.ingestion.crawler import load_latest_manifest, load_sources
from sustech_rag.parsing.html_parser import parse_html
from sustech_rag.parsing.pdf_parser import parse_pdf

logger = get_logger(__name__)


YEAR_RE = re.compile(r"(20[0-3]\d)\s*级?|\b(20[0-3]\d)\b")


def infer_effective_year(title: str, text: str) -> int | None:
    haystack = f"{title}\n{text[:2000]}"
    years = []
    for match in YEAR_RE.finditer(haystack):
        year = match.group(1) or match.group(2)
        if year:
            years.append(int(year))
    return max(years) if years else None


def pdf_title_from_url(url: str, fallback: str) -> str:
    name = unquote(Path(urlparse(url).path).stem).strip()
    if name:
        return name
    return fallback


def parse_manifest_rows(rows: list[RawManifest] | None = None) -> list[Document]:
    paths = load_paths()
    rows = rows or load_latest_manifest()
    _, source_configs = load_sources(Path(paths["project_root"]) / "configs" / "sources.yaml")
    source_by_id = {source.source_id: source for source in source_configs}
    documents: list[Document] = []
    for row in rows:
        local_path = Path(row.local_path)
        try:
            if "pdf" in row.mime_type.lower() or local_path.suffix.lower() == ".pdf":
                title, text, metadata = parse_pdf(local_path)
                title = pdf_title_from_url(row.url, row.title or title)
                source_type = "pdf"
            else:
                title, text, metadata = parse_html(local_path)
                source_type = "html"
        except Exception as exc:
            logger.warning("Parse failed path=%s error=%s", local_path, exc)
            continue
        if len(text) < 100:
            logger.info("Skipping short parsed document url=%s chars=%s", row.url, len(text))
            continue
        if any(marker in text[:1200] for marker in ("统一身份认证", "统一认证", "CAS", "用户登录")):
            logger.info("Skipping login-like parsed document url=%s", row.url)
            continue
        text_hash = sha256_text(text)
        year = infer_effective_year(title, text)
        source = source_by_id.get(row.source_id)
        documents.append(
            Document(
                doc_id=row.doc_id,
                title=title,
                source_url=row.url,
                source_type=source_type,
                category=source.category if source else metadata.get("category", "unknown"),
                authority_level=source.authority_level if source else metadata.get("authority_level", 50),
                effective_year=year,
                status="unknown",
                hash=text_hash,
                local_path=str(local_path),
                text=text,
                metadata={
                    **metadata,
                    "source_id": row.source_id,
                    "source_name": source.name if source else None,
                    "content_hash": row.content_hash,
                },
            )
        )
    out_dir = Path(paths["data"]["normalized"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
    logger.info("Wrote %s normalized documents to %s", len(documents), out_path)
    return documents


def load_documents() -> list[Document]:
    paths = load_paths()
    path = Path(paths["data"]["normalized"]) / "documents.jsonl"
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    return docs
