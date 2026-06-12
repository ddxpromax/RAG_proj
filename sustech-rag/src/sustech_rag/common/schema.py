from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceConfig(BaseModel):
    source_id: str
    name: str
    base_url: str
    seed_urls: list[str]
    allowed_domains: list[str]
    excluded_patterns: list[str] = Field(default_factory=list)
    category: str
    authority_level: int = 50
    crawl_depth: int = 2
    max_pages: int = 20
    enabled: bool = True


class RawManifest(BaseModel):
    doc_id: str
    url: str
    local_path: str
    mime_type: str
    status_code: int
    content_hash: str
    crawl_time: datetime
    source_id: str
    title: str | None = None
    parent_url: str | None = None


class Document(BaseModel):
    doc_id: str
    title: str
    source_url: str
    source_type: Literal["html", "pdf", "manual"] = "html"
    category: str
    department: str | None = None
    authority_level: int = 50
    published_date: str | None = None
    effective_year: int | None = None
    status: Literal["current", "historical", "superseded", "unknown"] = "unknown"
    hash: str
    local_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    chunk_index: int
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    text: str
    embedding_text: str
    display_text: str
    token_count: int
    hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    title: str | None = None
    url: HttpUrl | str | None = None
    score: float
    rank: int
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    evidence_status: Literal["supported", "partially_supported", "insufficient_evidence"]
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str
    retrieval_mode: str
    trace: dict[str, Any] = Field(default_factory=dict)

