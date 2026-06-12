from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sustech_rag.common.config import load_paths
from sustech_rag.common.schema import Document
from sustech_rag.common.utils import sha256_text
from sustech_rag.parsing.pipeline import infer_effective_year, load_documents


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_zs(limit: int) -> list[Document]:
    root = "https://zs.sustech.edu.cn/api/www/v1"
    headers = {"app-alias": "13b9222d-36dc-42c5-9954-16ae1403e8b9"}
    docs: list[Document] = []
    with httpx.Client(timeout=30, headers=headers) as client:
        list_resp = client.get(f"{root}/article/list", params={"page": 1, "limit": limit})
        list_resp.raise_for_status()
        items = list_resp.json().get("data", {}).get("items", [])
        for item in items:
            view = client.get(f"{root}/article/view", params={"alias": item["alias"]})
            data = view.json().get("data") or {}
            text = html_to_text(data.get("content") or "") or data.get("intro") or ""
            if len(text) < 80:
                continue
            title = data.get("title") or item.get("title") or "本科招生资料"
            doc_id = "api:zs:" + sha256_text(data.get("alias", title))
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    source_url=f"https://zs.sustech.edu.cn/#/detail?alias={data.get('alias')}",
                    source_type="html",
                    category="admissions",
                    department="本科招生",
                    authority_level=95,
                    published_date=data.get("published_date"),
                    effective_year=infer_effective_year(title, text),
                    status="current",
                    hash=sha256_text(text),
                    local_path=f"api://zs/{data.get('alias')}",
                    text=text,
                    metadata={"source_id": "admissions_undergrad", "source_name": "本科招生", "api_source": True},
                )
            )
    return docs


def fetch_gs(limit: int) -> list[Document]:
    root = "https://gs.sustech.edu.cn/api/www/v1"
    docs: list[Document] = []
    with httpx.Client(timeout=30) as client:
        list_resp = client.get(f"{root}/article/list", params={"page": 1, "limit": limit})
        list_resp.raise_for_status()
        items = list_resp.json().get("data", {}).get("items", [])
        for item in items:
            view = client.get(f"{root}/article/view", params={"alias": item["alias"]})
            data = view.json().get("data") or {}
            text = html_to_text(data.get("content") or "") or data.get("intro") or ""
            if len(text) < 80:
                continue
            title = data.get("title") or item.get("title") or "研究生院资料"
            doc_id = "api:gs:" + sha256_text(str(data.get("id") or data.get("alias") or title))
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    source_url=f"https://gs.sustech.edu.cn/#/article/view?alias={data.get('alias')}",
                    source_type="html",
                    category="graduate",
                    department=data.get("department_title") or "研究生院",
                    authority_level=90,
                    published_date=data.get("published_at") or data.get("published_date"),
                    effective_year=infer_effective_year(title, text),
                    status="current",
                    hash=sha256_text(text),
                    local_path=f"api://gs/{data.get('alias')}",
                    text=text,
                    metadata={"source_id": "graduate_school", "source_name": "研究生院", "api_source": True},
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zs-limit", type=int, default=35)
    parser.add_argument("--gs-limit", type=int, default=35)
    args = parser.parse_args()
    paths = load_paths()
    existing = load_documents()
    by_id = {doc.doc_id: doc for doc in existing}
    api_docs = fetch_zs(args.zs_limit) + fetch_gs(args.gs_limit)
    for doc in api_docs:
        by_id[doc.doc_id] = doc
    out_path = Path(paths["data"]["normalized"]) / "documents.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for doc in by_id.values():
            f.write(doc.model_dump_json() + "\n")
    print(f"api_docs={len(api_docs)} total_documents={len(by_id)} path={out_path}")


if __name__ == "__main__":
    main()
