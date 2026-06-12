from sustech_rag.chunking.chunker import chunk_document
from sustech_rag.common.schema import Document


def test_chunk_document_keeps_metadata() -> None:
    doc = Document(
        doc_id="doc1",
        title="测试文档",
        source_url="https://example.com",
        category="test",
        hash="h",
        local_path="/tmp/x",
        text="第一章 总则\n" + "这是正文。" * 120,
        effective_year=2024,
    )
    chunks = chunk_document(doc, max_chars=120, overlap=10)
    assert chunks
    assert chunks[0].metadata["effective_year"] == 2024
    assert "测试文档" in chunks[0].embedding_text

