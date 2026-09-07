from __future__ import annotations

import json
from pathlib import Path

import pytest

from devassist.domain import KnowledgeDocument
from devassist.ingestion import (
    ChunkingConfig,
    DuplicateDocumentIdError,
    IngestionPipeline,
    JsonlError,
    chunk_document,
    clean_document,
    ingest_jsonl,
    load_jsonl,
    normalize_markdown,
    write_jsonl,
)


def raw_document(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "project": "PyTorch",
        "version": "2.4",
        "source_type": "Documentation",
        "title": "CUDA 不可用",
        "content": "检查驱动版本。",
        "summary": "排查 CUDA",
        "resolution": ["更新驱动"],
        "url": "https://pytorch.org/docs/stable/notes/cuda.html",
        "authority": 1.0,
        "tags": ["CUDA", "Windows"],
    }
    value.update(overrides)
    return value


def test_utf8_jsonl_round_trip_is_deterministic(tmp_path: Path) -> None:
    target = tmp_path / "知识库.jsonl"
    records = [raw_document(content="中文内容: 显卡驱动。")]

    write_jsonl(target, records)

    assert load_jsonl(target) == records
    payload = target.read_bytes()
    assert "中文内容".encode() in payload
    assert b"\r\n" not in payload


def test_loader_accepts_windows_newlines_and_bom(tmp_path: Path) -> None:
    target = tmp_path / "windows.jsonl"
    first = json.dumps(raw_document(), ensure_ascii=False)
    second = json.dumps(raw_document(title="第二条"), ensure_ascii=False)
    target.write_bytes(("\ufeff" + first + "\r\n\r\n" + second + "\r\n").encode())

    loaded = load_jsonl(target)

    assert [item["title"] for item in loaded] == ["CUDA 不可用", "第二条"]


def test_loader_reports_bad_json_line(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text('{}\n{"broken":\n', encoding="utf-8")

    with pytest.raises(JsonlError, match=r"line 2"):
        load_jsonl(target)


def test_markdown_cleaning_preserves_fenced_code() -> None:
    markdown = (
        "\r\n说明文字   \r\n\r\n```python\r\n"
        "if torch.cuda.is_available():  \r\n\tprint('GPU')\r\n\r\n```\r\n\r\n"
    )

    cleaned = normalize_markdown(markdown)

    assert cleaned.startswith("说明文字")
    assert "if torch.cuda.is_available():  \n\tprint('GPU')\n\n```" in cleaned
    assert "\r" not in cleaned


def test_generated_document_id_is_stable_after_line_ending_normalization() -> None:
    first = clean_document(raw_document(id="", content="第一行\r\n第二行"))
    second = clean_document(raw_document(content="第一行\n第二行"))

    assert first.id == second.id
    assert first.id.startswith("doc_")
    assert first.project == "pytorch"
    assert first.tags == ("cuda", "windows")


def test_cleaner_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match=r"project.*version"):
        clean_document({})


def test_chunker_keeps_fenced_block_intact_and_has_stable_ids() -> None:
    code_lines = "\n".join(f"    value_{index} = {index}" for index in range(20))
    code = f"```python\n{code_lines}\n```"
    content = "A" * 90 + "\n\n" + code + "\n\n" + "B" * 90
    document = clean_document(raw_document(id="cuda-guide", content=content))
    config = ChunkingConfig(max_chars=100, overlap_chars=10)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert len(first) >= 3
    assert [item.id for item in first] == [item.id for item in second]
    code_chunks = [item.content for item in first if "```python" in item.content]
    assert code_chunks == [code]
    assert all(item.id.startswith("cuda-guide--chunk-") for item in first)


def test_chunker_keeps_prose_below_soft_limit() -> None:
    document = clean_document(raw_document(id="prose", content=("word " * 150).strip()))
    config = ChunkingConfig(max_chars=80, overlap_chars=20)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= config.max_chars for chunk in chunks)


def test_pipeline_deduplicates_normalized_documents() -> None:
    first = raw_document(id="first", content="同一内容\r\n第二行")
    duplicate = raw_document(id="second", content="同一内容\n第二行")

    result = IngestionPipeline().process([first, duplicate])

    assert result.stats.input_records == 2
    assert result.stats.clean_documents == 1
    assert result.stats.duplicate_documents == 1
    assert result.stats.output_chunks == 1
    assert result.documents[0].id == "first"


def test_pipeline_rejects_one_id_for_different_documents() -> None:
    with pytest.raises(DuplicateDocumentIdError, match="same-id"):
        IngestionPipeline().process(
            [
                raw_document(id="same-id", content="内容 A"),
                raw_document(id="same-id", content="内容 B"),
            ]
        )


def test_file_pipeline_writes_reloadable_domain_documents(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    destination = tmp_path / "nested" / "chunks.jsonl"
    write_jsonl(source, [raw_document(id="doc-1", content="内容 " * 80)])

    result = ingest_jsonl(
        source,
        destination,
        chunking=ChunkingConfig(max_chars=80, overlap_chars=8),
    )
    reloaded = [KnowledgeDocument.from_mapping(item) for item in load_jsonl(destination)]

    assert destination.exists()
    assert len(reloaded) == result.stats.output_chunks
    assert [item.id for item in reloaded] == [item.id for item in result.documents]
