from __future__ import annotations

import asyncio
import json
import mimetypes
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup

from sustech_rag.common.config import load_paths
from sustech_rag.common.logging import get_logger
from sustech_rag.common.schema import RawManifest, SourceConfig
from sustech_rag.common.utils import normalize_url, safe_filename, sha256_bytes

logger = get_logger(__name__)


def load_sources(config_path: str | Path) -> tuple[dict, list[SourceConfig]]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    sources = [SourceConfig(**item) for item in raw.get("sources", []) if item.get("enabled", True)]
    return defaults, sources


def allowed_url(url: str, source: SourceConfig) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in source.allowed_domains:
        return False
    return not any(pattern and pattern in parsed.path for pattern in source.excluded_patterns)


def is_pdf_url(url: str, content_type: str | None = None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return urlparse(url).path.lower().endswith(".pdf")


def is_html_response(url: str, content_type: str | None = None) -> bool:
    if content_type and any(marker in content_type.lower() for marker in ("text/html", "application/xhtml")):
        return True
    suffix = urlparse(url).path.lower().rsplit(".", 1)[-1]
    return "." not in urlparse(url).path or suffix in {"html", "htm", "php", "asp", "aspx"}


def looks_binary(data: bytes) -> bool:
    sample = data[:2048]
    return b"\x00" in sample


def discover_links(html: bytes, base_url: str, source: SourceConfig) -> list[str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.warning("Link discovery failed url=%s error=%s", base_url, exc)
        return []
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = normalize_url(tag["href"], base_url)
        if allowed_url(href, source):
            links.append(href)
    return links


def guess_extension(url: str, content_type: str | None) -> str:
    if is_pdf_url(url, content_type):
        return ".pdf"
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".html"


async def crawl(config_path: str | Path, limit_sources: list[str] | None = None) -> list[RawManifest]:
    paths = load_paths()
    defaults, sources = load_sources(config_path)
    if limit_sources:
        wanted = set(limit_sources)
        sources = [s for s in sources if s.source_id in wanted]

    raw_html = Path(paths["data"]["raw_html"])
    raw_pdf = Path(paths["data"]["raw_pdf"])
    manifest_dir = Path(paths["data"]["manifests"])
    raw_html.mkdir(parents=True, exist_ok=True)
    raw_pdf.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": defaults.get("user_agent", "SUSTech-RAG-Course-Project/0.1")}
    timeout = httpx.Timeout(float(defaults.get("timeout_seconds", 30)))
    manifests: list[RawManifest] = []

    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for source in sources:
            queue = deque((normalize_url(url), 0, None) for url in source.seed_urls)
            seen: set[str] = set()
            saved = 0
            logger.info("Crawling source=%s max_pages=%s", source.source_id, source.max_pages)
            while queue and saved < source.max_pages:
                url, depth, parent = queue.popleft()
                if url in seen or not allowed_url(url, source):
                    continue
                seen.add(url)
                try:
                    response = await client.get(url)
                except Exception as exc:
                    logger.warning("Fetch failed url=%s error=%s", url, exc)
                    continue
                if response.status_code >= 400:
                    logger.warning("Skipping url=%s status=%s", url, response.status_code)
                    continue

                content_type = response.headers.get("content-type", "")
                body = response.content
                final_url = str(response.url)
                if not allowed_url(normalize_url(final_url), source):
                    logger.info("Skipping redirected out-of-scope url=%s from=%s", final_url, url)
                    continue
                is_pdf = is_pdf_url(final_url, content_type)
                is_html = is_html_response(final_url, content_type)
                if not is_pdf and (not is_html or looks_binary(body)):
                    logger.info("Skipping non HTML/PDF url=%s content_type=%s", final_url, content_type)
                    continue

                if is_pdf:
                    max_bytes = int(defaults.get("max_pdf_bytes", 52_428_800))
                    target_dir = raw_pdf / source.source_id
                else:
                    max_bytes = int(defaults.get("max_html_bytes", 5_242_880))
                    target_dir = raw_html / source.source_id
                if len(body) > max_bytes:
                    logger.warning("Skipping oversized url=%s bytes=%s", url, len(body))
                    continue

                content_hash = sha256_bytes(body)
                ext = guess_extension(str(response.url), content_type)
                filename = safe_filename(f"{saved:04d}_{content_hash[:16]}", ext)
                target_dir.mkdir(parents=True, exist_ok=True)
                local_path = target_dir / filename
                local_path.write_bytes(body)
                doc_id = f"sha256:{content_hash}"
                manifest = RawManifest(
                    doc_id=doc_id,
                    url=normalize_url(str(response.url)),
                    local_path=str(local_path),
                    mime_type=content_type or ("application/pdf" if ext == ".pdf" else "text/html"),
                    status_code=response.status_code,
                    content_hash=content_hash,
                    crawl_time=datetime.now(UTC),
                    source_id=source.source_id,
                    parent_url=parent,
                )
                manifests.append(manifest)
                saved += 1

                if depth < source.crawl_depth and is_html:
                    for link in discover_links(body, str(response.url), source):
                        if link not in seen and not re.search(r"\.(jpg|jpeg|png|gif|zip|rar)$", link, re.I):
                            queue.append((link, depth + 1, url))

    out_path = manifest_dir / f"raw_manifest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in manifests:
            f.write(item.model_dump_json() + "\n")
    latest = manifest_dir / "raw_manifest.latest.jsonl"
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Wrote %s raw manifest rows to %s", len(manifests), out_path)
    return manifests


def load_latest_manifest() -> list[RawManifest]:
    paths = load_paths()
    path = Path(paths["data"]["manifests"]) / "raw_manifest.latest.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(RawManifest(**json.loads(line)))
    return rows
