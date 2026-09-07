"""Markdown-aware deterministic chunking for knowledge documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from devassist.domain import KnowledgeDocument

_FENCE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based chunking limits.

    ``max_chars`` is a soft limit: an individual fenced code block is never split,
    even when that means emitting one larger chunk.
    """

    max_chars: int = 900
    overlap_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars < 32:
            raise ValueError("max_chars must be at least 32")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    is_code: bool


def _markdown_units(text: str) -> list[_Unit]:
    """Split markdown into prose paragraphs and complete fenced code blocks."""

    units: list[_Unit] = []
    current: list[str] = []
    in_fence = False
    fence_character = ""
    fence_width = 0

    def flush(*, is_code: bool) -> None:
        nonlocal current
        joined = "\n".join(current).strip("\n")
        if joined:
            units.append(_Unit(joined, is_code))
        current = []

    for line in text.split("\n"):
        match = _FENCE.match(line)
        if in_fence:
            current.append(line)
            if match is not None:
                marker = match.group("marker")
                if marker[0] == fence_character and len(marker) >= fence_width:
                    in_fence = False
                    flush(is_code=True)
            continue

        if match is not None:
            flush(is_code=False)
            marker = match.group("marker")
            fence_character = marker[0]
            fence_width = len(marker)
            in_fence = True
            current.append(line)
            continue

        if not line.strip():
            flush(is_code=False)
            continue
        current.append(line)

    flush(is_code=in_fence)
    return units


def _split_large_prose(text: str, config: ChunkingConfig) -> list[str]:
    if len(text) <= config.max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        upper = min(start + config.max_chars, text_length)
        end = upper
        if upper < text_length:
            window = text[start:upper]
            minimum_break = config.max_chars // 2
            candidates = (
                window.rfind("\n"),
                window.rfind("。"),
                window.rfind(". "),
                window.rfind(" "),
            )
            best_break = max(candidates)
            if best_break >= minimum_break:
                end = start + best_break + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= text_length:
            break
        next_start = max(start + 1, end - config.overlap_chars)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return pieces


def _sized_units(text: str, config: ChunkingConfig) -> list[_Unit]:
    sized: list[_Unit] = []
    for unit in _markdown_units(text):
        if unit.is_code or len(unit.text) <= config.max_chars:
            sized.append(unit)
            continue
        sized.extend(_Unit(piece, False) for piece in _split_large_prose(unit.text, config))
    return sized


def _pack_units(units: Iterable[_Unit], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current: list[_Unit] = []

    def render(values: list[_Unit]) -> str:
        return "\n\n".join(value.text for value in values).strip()

    for unit in units:
        if len(unit.text) > config.max_chars:
            if current:
                chunks.append(render(current))
                current = []
            chunks.append(unit.text)
            continue

        candidate = render([*current, unit])
        if current and len(candidate) > config.max_chars:
            chunks.append(render(current))
            carry: list[_Unit] = []
            carry_size = 0
            for previous in reversed(current):
                addition = len(previous.text) + (2 if carry else 0)
                candidate_size = carry_size + addition
                joined_size = candidate_size + len(unit.text) + (2 if candidate_size else 0)
                if (
                    previous.is_code
                    or candidate_size > config.overlap_chars
                    or joined_size > config.max_chars
                ):
                    break
                carry.insert(0, previous)
                carry_size = candidate_size
            current = carry
        current.append(unit)

    if current:
        rendered = render(current)
        if not chunks or chunks[-1] != rendered:
            chunks.append(rendered)
    return chunks


def _chunk_id(document_id: str, index: int, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}--chunk-{index:04d}-{digest}"


def chunk_document(
    document: KnowledgeDocument,
    config: ChunkingConfig | None = None,
) -> tuple[KnowledgeDocument, ...]:
    """Split one document into stable, domain-compatible chunks."""

    selected_config = config or ChunkingConfig()
    if len(document.content) <= selected_config.max_chars:
        return (document,)

    contents = _pack_units(_sized_units(document.content, selected_config), selected_config)
    if len(contents) == 1 and contents[0] == document.content:
        return (document,)
    return tuple(
        replace(document, id=_chunk_id(document.id, index, content), content=content)
        for index, content in enumerate(contents, start=1)
    )


def chunk_documents(
    documents: Iterable[KnowledgeDocument],
    config: ChunkingConfig | None = None,
) -> tuple[KnowledgeDocument, ...]:
    """Chunk documents in input order."""

    selected_config = config or ChunkingConfig()
    return tuple(
        chunk for document in documents for chunk in chunk_document(document, selected_config)
    )
