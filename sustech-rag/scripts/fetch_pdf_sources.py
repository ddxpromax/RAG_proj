from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import PROJECT_ROOT, load_paths
from sustech_rag.common.schema import RawManifest
from sustech_rag.common.utils import normalize_url, safe_filename, sha256_bytes


def load_source(source_id: str) -> tuple[dict, dict]:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8"))
    defaults = config.get("defaults", {})
    for source in config.get("sources", []):
        if source["source_id"] == source_id:
            return defaults, source
    raise ValueError(f"Unknown source_id: {source_id}")


def read_existing_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_pdfs(directory_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for tag in soup.find_all("a", href=True):
        url = normalize_url(tag["href"], directory_url)
        if url.lower().split("?", 1)[0].endswith(".pdf"):
            urls.append(url)
    return list(dict.fromkeys(urls))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="undergrad_training_plans_2024")
    parser.add_argument("--limit", type=int, default=45)
    args = parser.parse_args()

    paths = load_paths()
    defaults, source = load_source(args.source)
    raw_pdf = Path(paths["data"]["raw_pdf"]) / source["source_id"]
    manifest_dir = Path(paths["data"]["manifests"])
    raw_pdf.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    latest = manifest_dir / "raw_manifest.latest.jsonl"
    existing = read_existing_manifest(latest)
    seen_hashes = {row.get("content_hash") for row in existing}
    seen_urls = {row.get("url") for row in existing}

    headers = {"User-Agent": defaults.get("user_agent", "SUSTech-RAG-Course-Project/0.1")}
    timeout = httpx.Timeout(float(defaults.get("timeout_seconds", 30)))
    new_rows: list[dict] = []
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        pdf_urls: list[str] = []
        for seed in source["seed_urls"]:
            response = client.get(seed)
            response.raise_for_status()
            pdf_urls.extend(discover_pdfs(str(response.url), response.text))
        for index, url in enumerate(list(dict.fromkeys(pdf_urls))[: args.limit]):
            if url in seen_urls:
                continue
            response = client.get(url)
            response.raise_for_status()
            body = response.content
            content_hash = sha256_bytes(body)
            if content_hash in seen_hashes:
                continue
            filename = safe_filename(f"{len(existing) + len(new_rows):04d}_{content_hash[:16]}", ".pdf")
            local_path = raw_pdf / filename
            local_path.write_bytes(body)
            title = Path(url.split("/", 1)[-1]).name
            row = RawManifest(
                doc_id=f"sha256:{content_hash}",
                url=normalize_url(str(response.url)),
                local_path=str(local_path),
                mime_type=response.headers.get("content-type") or "application/pdf",
                status_code=response.status_code,
                content_hash=content_hash,
                crawl_time=datetime.now(UTC),
                source_id=source["source_id"],
                title=title,
                parent_url=source["seed_urls"][0],
            ).model_dump(mode="json")
            new_rows.append(row)
            seen_hashes.add(content_hash)
            seen_urls.add(url)

    out_path = manifest_dir / f"raw_manifest_pdf_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    combined = existing + new_rows
    with out_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"discovered={len(pdf_urls)} added={len(new_rows)} total_manifest={len(combined)}")
    print(out_path)


if __name__ == "__main__":
    main()
