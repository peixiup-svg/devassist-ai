"""Internal immutable domain objects used by retrieval and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    id: str
    project: str
    version: str
    source_type: str
    title: str
    content: str
    summary: str
    resolution: tuple[str, ...]
    url: str
    authority: float = 1.0
    tags: tuple[str, ...] = ()
    published_at: str | None = None

    @property
    def searchable_text(self) -> str:
        tags = " ".join(self.tags)
        return f"{self.project} {self.version} {self.title} {tags} {self.content} {self.summary}"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> KnowledgeDocument:
        required = {
            "id",
            "project",
            "version",
            "source_type",
            "title",
            "content",
            "summary",
            "resolution",
            "url",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"document is missing required fields: {', '.join(missing)}")
        authority = float(value.get("authority", 1.0))
        if not 0.0 <= authority <= 1.0:
            raise ValueError("document authority must be between 0 and 1")
        return cls(
            id=str(value["id"]).strip(),
            project=str(value["project"]).strip().lower(),
            version=str(value["version"]).strip().lower(),
            source_type=str(value["source_type"]).strip().lower(),
            title=str(value["title"]).strip(),
            content=str(value["content"]).strip(),
            summary=str(value["summary"]).strip(),
            resolution=tuple(
                str(item).strip() for item in value["resolution"] if str(item).strip()
            ),
            url=str(value["url"]).strip(),
            authority=authority,
            tags=tuple(str(item).strip().lower() for item in value.get("tags", [])),
            published_at=(str(value["published_at"]) if value.get("published_at") else None),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: KnowledgeDocument
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    clean_query: str
    project: str | None
    version: str | None
    intent: str
    exception_names: tuple[str, ...]
    source_types: tuple[str, ...]
    missing_information: tuple[str, ...]
    safety_warnings: tuple[str, ...]
