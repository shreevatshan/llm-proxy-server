"""Backend-agnostic search result types and the model-facing result format.

Shared by every search backend (SearXNG, 4get) so the model sees identical
tool-result text regardless of which one answered.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    date: Optional[str] = None


@dataclass(frozen=True)
class SearchSucceeded:
    query: str
    results: List[SearchResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class SearchFailed:
    query: str
    error_code: str  # too_many_requests | unavailable | invalid_tool_input | query_too_long
    message: str

    @property
    def ok(self) -> bool:
        return False


SearchOutcome = Union[SearchSucceeded, SearchFailed]

MAX_QUERY_CHARS = 500


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def format_outcome_text(outcome: SearchOutcome) -> str:
    """Text the model sees as the tool result (LiteLLM's Title/URL/Snippet blocks)."""
    if not outcome.ok:
        return f"Search failed: {outcome.message}"
    if not outcome.results:
        return f"No results found for: {outcome.query}"
    blocks = []
    for r in outcome.results:
        lines = [f"Title: {r.title}", f"URL: {r.url}"]
        if r.date:
            lines.append(f"Date: {r.date}")
        lines.append(f"Snippet: {r.snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
