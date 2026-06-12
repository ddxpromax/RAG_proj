from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"spm", "from", "source", "ref", "fbclid", "gclid"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def safe_filename(value: str, suffix: str = "") -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if len(base) > 120:
        base = base[:120]
    return f"{base or 'item'}{suffix}"


def normalize_url(url: str, base_url: str | None = None) -> str:
    if base_url:
        url = urljoin(base_url, url)
    parsed = urlparse(url)
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key in TRACKING_QUERY_KEYS or any(key.startswith(p) for p in TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, value))
    query = urlencode(query_items, doseq=True)
    path = parsed.path or "/"
    normalized = parsed._replace(fragment="", query=query, path=path)
    return urlunparse(normalized)


def write_text(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    import json

    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

