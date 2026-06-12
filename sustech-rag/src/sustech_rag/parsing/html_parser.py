from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html(path: str | Path) -> tuple[str, str, dict]:
    html = Path(path).read_bytes()
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="txt",
        )
        if extracted and len(extracted.strip()) > 120:
            soup_for_title = BeautifulSoup(html, "html.parser")
            title = ""
            if soup_for_title.title and soup_for_title.title.string:
                title = soup_for_title.title.string.strip()
            heading = soup_for_title.find(["h1", "h2"])
            if heading and heading.get_text(strip=True):
                title = heading.get_text(" ", strip=True)
            return title or Path(path).stem, _clean_text(extracted), {"parser": "trafilatura"}
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    heading = soup.find(["h1", "h2"])
    if heading and heading.get_text(strip=True):
        title = heading.get_text(" ", strip=True)
    main = soup.find("main") or soup.find("article") or soup.body or soup
    parts: list[str] = []
    for node in main.find_all(["h1", "h2", "h3", "p", "li", "td", "th"], recursive=True):
        text = node.get_text(" ", strip=True)
        if text:
            if node.name in {"h1", "h2", "h3"}:
                parts.append("\n" + text + "\n")
            else:
                parts.append(text)
    if not parts:
        parts = [main.get_text("\n", strip=True)]
    metadata = {
        "parser": "beautifulsoup",
        "canonical": (soup.find("link", rel="canonical") or {}).get("href"),
    }
    return title or "Untitled HTML document", _clean_text("\n".join(parts)), metadata
