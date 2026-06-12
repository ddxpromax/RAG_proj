from __future__ import annotations

from pathlib import Path


def parse_pdf(path: str | Path) -> tuple[str, str, dict]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("PyMuPDF is required for PDF parsing. Install `.[full]`.") from exc

    doc = fitz.open(str(path))
    pages: list[str] = []
    for index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"\n[PDF_PAGE {index}]\n{text}")
    title = doc.metadata.get("title") or Path(path).stem
    metadata = {"parser": "pymupdf", "page_count": doc.page_count}
    return title, "\n".join(pages).strip(), metadata

