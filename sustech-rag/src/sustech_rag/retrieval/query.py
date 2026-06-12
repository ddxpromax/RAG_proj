from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    original_query: str
    standalone_query: str
    entities: list[str] = field(default_factory=list)
    effective_year: int | None = None
    intent: str = "fact_lookup"
    sub_queries: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)


YEAR_RE = re.compile(r"(20[0-3]\d)\s*级?")


def analyze_query(query: str, history: list[dict] | None = None) -> QueryAnalysis:
    history = history or []
    year_match = YEAR_RE.search(query)
    effective_year = int(year_match.group(1)) if year_match else None
    intent = "comparison" if any(word in query for word in ("比较", "哪个", "更高", "区别", "差异")) else "fact_lookup"
    filters = {}
    if effective_year:
        filters["effective_year"] = effective_year
    sub_queries: list[str] = []
    if intent == "comparison":
        parts = re.split(r"和|与|跟|、", query, maxsplit=1)
        if len(parts) == 2:
            left = parts[0].strip("？?，, ")
            right = parts[1].strip("？?，, ")
            sub_queries = [left, right]
    return QueryAnalysis(
        original_query=query,
        standalone_query=query,
        effective_year=effective_year,
        intent=intent,
        sub_queries=sub_queries,
        filters=filters,
    )

