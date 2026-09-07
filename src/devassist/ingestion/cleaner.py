"""Normalization, stable identifiers, and fingerprints for knowledge records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from devassist.domain import KnowledgeDocument

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")
_TRAILING_HORIZONTAL_SPACE = re.compile(r"[ \t]+$")
_INLINE_SPACE = re.compile(r"\s+")


def normalize_inline(value: object) -> str:
    """Normalize a metadata value to one trimmed line."""

    text = _CONTROL_CHARACTERS.sub("", str(value))
    return _INLINE_SPACE.sub(" ", text).strip()


def normalize_markdown(value: object) -> str:
    """Normalize line endings and prose while preserving fenced code contents.

    Indentation, tabs, blank lines, and trailing whitespace *inside* a code fence are
    retained. Outside fences only control characters, trailing horizontal whitespace,
    and runs of more than two blank lines are removed.
    """

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    output: list[str] = []
    in_fence = False
    fence_character = ""
    fence_width = 0
    blank_run = 0

    for original_line in text.split("\n"):
        match = _FENCE.match(original_line)
        if match is not None:
            marker = match.group("marker")
            if not in_fence:
                in_fence = True
                fence_character = marker[0]
                fence_width = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_width:
                in_fence = False
            output.append(original_line if in_fence else original_line.rstrip())
            blank_run = 0
            continue

        if in_fence:
            output.append(original_line)
            continue

        cleaned = _CONTROL_CHARACTERS.sub("", original_line)
        cleaned = _TRAILING_HORIZONTAL_SPACE.sub("", cleaned)
        if not cleaned.strip():
            blank_run += 1
            if blank_run <= 2:
                output.append("")
            continue
        blank_run = 0
        output.append(cleaned)

    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    return "\n".join(output)


def _string_sequence(value: object, *, lower: bool = False) -> list[str]:
    if value is None:
        return []
    candidates: Sequence[object]
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, Sequence):
        candidates = value
    else:
        raise ValueError("expected a string or a sequence of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        clean_item = normalize_inline(item)
        if lower:
            clean_item = clean_item.lower()
        if clean_item and clean_item not in seen:
            normalized.append(clean_item)
            seen.add(clean_item)
    return normalized


def stable_document_id(value: Mapping[str, Any]) -> str:
    """Build a repeatable content-derived identifier for a normalized mapping."""

    identity = {
        "content": normalize_markdown(value.get("content", "")),
        "project": normalize_inline(value.get("project", "")).lower(),
        "source_type": normalize_inline(value.get("source_type", "")).lower(),
        "title": normalize_inline(value.get("title", "")),
        "url": normalize_inline(value.get("url", "")),
        "version": normalize_inline(value.get("version", "")).lower(),
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"doc_{digest}"


def clean_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical mapping accepted by ``KnowledgeDocument``."""

    clean_value: dict[str, Any] = {
        "project": normalize_inline(value.get("project", "")).lower(),
        "version": normalize_inline(value.get("version", "")).lower(),
        "source_type": normalize_inline(value.get("source_type", "")).lower(),
        "title": normalize_inline(value.get("title", "")),
        "content": normalize_markdown(value.get("content", "")),
        "summary": normalize_inline(value.get("summary", "")),
        "resolution": _string_sequence(value.get("resolution", [])),
        "url": normalize_inline(value.get("url", "")),
        "authority": value.get("authority", 1.0),
        "tags": _string_sequence(value.get("tags", []), lower=True),
        "published_at": normalize_inline(value["published_at"])
        if value.get("published_at")
        else None,
    }
    required_non_empty = ("project", "version", "source_type", "title", "content", "url")
    empty_fields = [field for field in required_non_empty if not clean_value[field]]
    if empty_fields:
        raise ValueError(f"document fields cannot be empty: {', '.join(empty_fields)}")
    supplied_id = normalize_inline(value.get("id", ""))
    clean_value["id"] = supplied_id or stable_document_id(clean_value)
    return clean_value


def clean_document(value: Mapping[str, Any] | KnowledgeDocument) -> KnowledgeDocument:
    """Normalize and validate a mapping or an existing domain document."""

    raw_value: Mapping[str, Any] = asdict(value) if isinstance(value, KnowledgeDocument) else value
    return KnowledgeDocument.from_mapping(clean_mapping(raw_value))


def document_fingerprint(document: KnowledgeDocument) -> str:
    """Return a stable semantic fingerprint used to remove duplicate sources."""

    identity = {
        "content": document.content,
        "project": document.project,
        "source_type": document.source_type,
        "title": document.title,
        "version": document.version,
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def document_to_mapping(document: KnowledgeDocument) -> dict[str, Any]:
    """Convert a domain document into a JSON-compatible canonical mapping."""

    return {
        "id": document.id,
        "project": document.project,
        "version": document.version,
        "source_type": document.source_type,
        "title": document.title,
        "content": document.content,
        "summary": document.summary,
        "resolution": list(document.resolution),
        "url": document.url,
        "authority": document.authority,
        "tags": list(document.tags),
        "published_at": document.published_at,
    }
