"""Public API for DevAssist's offline ingestion pipeline."""

from devassist.ingestion.chunker import ChunkingConfig, chunk_document, chunk_documents
from devassist.ingestion.cleaner import (
    clean_document,
    clean_mapping,
    document_fingerprint,
    document_to_mapping,
    normalize_inline,
    normalize_markdown,
    stable_document_id,
)
from devassist.ingestion.loader import JsonlError, iter_jsonl, load_jsonl, write_jsonl
from devassist.ingestion.pipeline import (
    DuplicateDocumentIdError,
    IngestionPipeline,
    IngestionResult,
    IngestionStats,
    ingest_jsonl,
)

__all__ = [
    "ChunkingConfig",
    "DuplicateDocumentIdError",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionStats",
    "JsonlError",
    "chunk_document",
    "chunk_documents",
    "clean_document",
    "clean_mapping",
    "document_fingerprint",
    "document_to_mapping",
    "ingest_jsonl",
    "iter_jsonl",
    "load_jsonl",
    "normalize_inline",
    "normalize_markdown",
    "stable_document_id",
    "write_jsonl",
]
