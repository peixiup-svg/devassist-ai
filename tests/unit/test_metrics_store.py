from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from devassist.domain import KnowledgeDocument
from devassist.metrics import AppMetrics
from devassist.store import KnowledgeStore


def test_metrics_empty_snapshot_is_zeroed() -> None:
    snapshot = AppMetrics().snapshot()

    assert snapshot.requests == 0
    assert snapshot.answers == 0
    assert snapshot.refusals == 0
    assert snapshot.feedback_items == 0
    assert snapshot.p50_latency_ms == 0.0
    assert snapshot.p95_latency_ms == 0.0


def test_metrics_counts_outcomes_percentiles_and_bounded_history() -> None:
    metrics = AppMetrics(history_size=3)
    metrics.record_diagnosis(answered=True, latency_ms=10.12349)
    metrics.record_diagnosis(answered=False, latency_ms=20.0)
    metrics.record_diagnosis(answered=True, latency_ms=30.0)
    metrics.record_diagnosis(answered=False, latency_ms=40.0)
    metrics.record_feedback()
    metrics.record_feedback()

    snapshot = metrics.snapshot()

    assert snapshot.requests == 4
    assert snapshot.answers == 2
    assert snapshot.refusals == 2
    assert snapshot.feedback_items == 2
    assert snapshot.p50_latency_ms == 30.0
    assert snapshot.p95_latency_ms == 40.0


def test_metrics_updates_are_thread_safe() -> None:
    metrics = AppMetrics(history_size=200)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: metrics.record_diagnosis(
                    answered=index % 2 == 0, latency_ms=float(index)
                ),
                range(100),
            )
        )

    snapshot = metrics.snapshot()
    assert (snapshot.requests, snapshot.answers, snapshot.refusals) == (100, 50, 50)


def test_store_loads_comments_blank_lines_and_indexes_by_id(
    tmp_path: Path,
    sample_documents: tuple[KnowledgeDocument, ...],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> None:
    path = write_knowledge_base(tmp_path / "knowledge.jsonl", sample_documents)
    original = path.read_text(encoding="utf-8")
    path.write_text("# fixture\n\n" + original, encoding="utf-8")
    store = KnowledgeStore(path)

    generation = store.load()

    assert generation == 1
    assert store.generation == 1
    assert store.documents == sample_documents
    assert store.get("pytorch-load-26") == sample_documents[0]
    assert store.get("missing") is None


def test_store_successful_reload_atomically_replaces_documents(
    tmp_path: Path,
    sample_documents: tuple[KnowledgeDocument, ...],
    document_factory: Callable[..., KnowledgeDocument],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> None:
    path = write_knowledge_base(tmp_path / "knowledge.jsonl", sample_documents[:1])
    store = KnowledgeStore(path)
    assert store.load() == 1
    replacement = document_factory(id="replacement")
    write_knowledge_base(path, (replacement,))

    assert store.load() == 2
    assert store.documents == (replacement,)
    assert store.get("pytorch-load-26") is None
    assert store.get("replacement") == replacement


def test_failed_reload_preserves_previous_generation(
    tmp_path: Path,
    sample_documents: tuple[KnowledgeDocument, ...],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> None:
    path = write_knowledge_base(tmp_path / "knowledge.jsonl", sample_documents[:1])
    store = KnowledgeStore(path)
    store.load()
    before = store.documents
    path.write_text('{"broken":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        store.load()

    assert store.generation == 1
    assert store.documents == before


def test_store_rejects_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="knowledge base not found"):
        KnowledgeStore(tmp_path / "missing.jsonl").load()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n# no documents\n", encoding="utf-8")
    with pytest.raises(ValueError, match="knowledge base is empty"):
        KnowledgeStore(empty).load()


def test_store_reports_json_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('# comment\n{"broken":\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":2:"):
        KnowledgeStore(path).load()


def test_store_rejects_empty_duplicate_and_non_http_ids(
    tmp_path: Path,
    document_factory: Callable[..., KnowledgeDocument],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> None:
    empty_id_path = write_knowledge_base(tmp_path / "empty-id.jsonl", (document_factory(id=""),))
    with pytest.raises(ValueError, match="empty document id"):
        KnowledgeStore(empty_id_path).load()

    duplicate = document_factory(id="duplicate")
    duplicate_path = write_knowledge_base(tmp_path / "duplicate.jsonl", (duplicate, duplicate))
    with pytest.raises(ValueError, match="duplicate document id"):
        KnowledgeStore(duplicate_path).load()

    bad_url_path = write_knowledge_base(
        tmp_path / "bad-url.jsonl",
        (document_factory(id="bad-url", url="file:///etc/passwd"),),
    )
    with pytest.raises(ValueError, match=r"URL must be http\(s\)"):
        KnowledgeStore(bad_url_path).load()


def test_store_readers_observe_consistent_snapshot_during_reload(
    tmp_path: Path,
    sample_documents: tuple[KnowledgeDocument, ...],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> None:
    path = write_knowledge_base(tmp_path / "knowledge.jsonl", sample_documents)
    store = KnowledgeStore(path)
    store.load()

    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(executor.map(lambda _index: store.documents, range(20)))

    assert all(snapshot == sample_documents for snapshot in snapshots)


def test_store_propagates_domain_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad-authority.jsonl"
    payload = {
        "id": "bad-authority",
        "project": "pytorch",
        "version": "2.6",
        "source_type": "documentation",
        "title": "title",
        "content": "content",
        "summary": "summary",
        "resolution": [],
        "url": "https://example.test",
        "authority": -0.1,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authority"):
        KnowledgeStore(path).load()
