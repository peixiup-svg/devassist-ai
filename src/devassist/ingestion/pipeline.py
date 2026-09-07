"""End-to-end deterministic ingestion for local JSONL knowledge bases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devassist.domain import KnowledgeDocument
from devassist.ingestion.chunker import ChunkingConfig, chunk_documents
from devassist.ingestion.cleaner import (
    clean_document,
    document_fingerprint,
    document_to_mapping,
)
from devassist.ingestion.loader import load_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class IngestionStats:
    input_records: int
    clean_documents: int
    duplicate_documents: int
    output_chunks: int


@dataclass(frozen=True, slots=True)
class IngestionResult:
    documents: tuple[KnowledgeDocument, ...]
    stats: IngestionStats


class DuplicateDocumentIdError(ValueError):
    """Raised when one identifier refers to two non-identical documents."""


@dataclass(frozen=True, slots=True)
class IngestionPipeline:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def process(
        self,
        records: Iterable[Mapping[str, Any] | KnowledgeDocument],
    ) -> IngestionResult:
        """Clean, validate, de-duplicate, and chunk in-memory records."""

        unique_documents: list[KnowledgeDocument] = []
        seen_fingerprints: set[str] = set()
        identifiers: dict[str, str] = {}
        input_count = 0
        duplicate_count = 0

        for record in records:
            input_count += 1
            document = clean_document(record)
            fingerprint = document_fingerprint(document)
            if fingerprint in seen_fingerprints:
                duplicate_count += 1
                continue
            previous_fingerprint = identifiers.get(document.id)
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                raise DuplicateDocumentIdError(
                    f"document id {document.id!r} refers to different normalized content"
                )
            identifiers[document.id] = fingerprint
            seen_fingerprints.add(fingerprint)
            unique_documents.append(document)

        chunks = chunk_documents(unique_documents, self.chunking)
        stats = IngestionStats(
            input_records=input_count,
            clean_documents=len(unique_documents),
            duplicate_documents=duplicate_count,
            output_chunks=len(chunks),
        )
        return IngestionResult(documents=chunks, stats=stats)

    def run(self, source: str | Path, destination: str | Path | None = None) -> IngestionResult:
        """Run the pipeline for a JSONL source and optionally persist its chunks."""

        result = self.process(load_jsonl(source))
        if destination is not None:
            write_jsonl(
                destination,
                (document_to_mapping(document) for document in result.documents),
            )
        return result


def ingest_jsonl(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    chunking: ChunkingConfig | None = None,
) -> IngestionResult:
    """Convenience wrapper for the common file-to-file ingestion workflow."""

    return IngestionPipeline(chunking=chunking or ChunkingConfig()).run(source, destination)
