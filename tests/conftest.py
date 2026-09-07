from __future__ import annotations

import ipaddress
import json
import socket
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from devassist.config import Settings
from devassist.domain import KnowledgeDocument


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail immediately if a test accidentally attempts real network I/O."""

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def is_loopback(address: object) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        host = str(address[0])
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def guarded_connect(instance: socket.socket, address: object) -> None:
        if not is_loopback(address):
            raise AssertionError(f"tests must not access external network address: {address!r}")
        original_connect(instance, address)  # type: ignore[arg-type]

    def guarded_create_connection(
        address: object, *args: object, **kwargs: object
    ) -> socket.socket:
        if not is_loopback(address):
            raise AssertionError(f"tests must not access external network address: {address!r}")
        return original_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield


@pytest.fixture
def document_factory() -> Callable[..., KnowledgeDocument]:
    def factory(**overrides: Any) -> KnowledgeDocument:
        values: dict[str, Any] = {
            "id": "pytorch-load-26",
            "project": "pytorch",
            "version": "2.6",
            "source_type": "changelog",
            "title": "PyTorch 2.6 torch.load weights_only change",
            "content": (
                "PyTorch 2.6 makes torch.load use weights_only=True by default. "
                "A trusted legacy checkpoint may be loaded with weights_only=False."
            ),
            "summary": "The weights_only default changed in PyTorch 2.6.",
            "resolution": (
                "Prefer loading a state_dict.",
                "Only disable weights_only for a trusted checkpoint.",
            ),
            "url": "https://docs.example.test/pytorch/2.6/torch-load",
            "authority": 1.0,
            "tags": ("torch.load", "weights_only"),
            "published_at": "2025-01-01",
        }
        values.update(overrides)
        return KnowledgeDocument(**values)

    return factory


@pytest.fixture
def sample_documents(
    document_factory: Callable[..., KnowledgeDocument],
) -> tuple[KnowledgeDocument, ...]:
    return (
        document_factory(),
        document_factory(
            id="pytorch-load-25",
            version="2.5",
            source_type="migration",
            title="PyTorch 2.5 torch.load behavior",
            content="PyTorch 2.5 uses weights_only=False by default for torch.load.",
            summary="PyTorch 2.5 retains the older torch.load default.",
            resolution=("Verify the installed torch version first.",),
            url="https://docs.example.test/pytorch/2.5/torch-load",
            tags=("torch.load", "weights_only", "2.5"),
        ),
        document_factory(
            id="transformers-padding",
            project="transformers",
            version="all",
            source_type="issue",
            title="Tokenizer has no pad_token",
            content=(
                "Asking to pad fails when the tokenizer has no padding token. "
                "For decoder-only inference, pad_token can use eos_token."
            ),
            summary="The tokenizer has no pad token.",
            resolution=("Set pad_token to eos_token for decoder-only inference.",),
            url="https://docs.example.test/transformers/padding",
            tags=("pad_token", "tokenizer"),
        ),
        document_factory(
            id="fastapi-validation",
            project="fastapi",
            version="all",
            source_type="documentation",
            title="FastAPI 422 validation errors",
            content="FastAPI returns 422 when a JSON request does not match its Pydantic model.",
            summary="The request body does not match the declared schema.",
            resolution=("Inspect detail.loc and compare the request with OpenAPI.",),
            url="https://docs.example.test/fastapi/validation",
            tags=("422", "pydantic"),
        ),
    )


def _document_payload(document: KnowledgeDocument) -> dict[str, Any]:
    payload = asdict(document)
    payload["resolution"] = list(document.resolution)
    payload["tags"] = list(document.tags)
    return payload


@pytest.fixture
def write_knowledge_base() -> Callable[[Path, tuple[KnowledgeDocument, ...]], Path]:
    def write(path: Path, documents: tuple[KnowledgeDocument, ...]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(_document_payload(document), ensure_ascii=False) for document in documents
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return write


@pytest.fixture
def settings_factory(
    tmp_path: Path,
    sample_documents: tuple[KnowledgeDocument, ...],
    write_knowledge_base: Callable[[Path, tuple[KnowledgeDocument, ...]], Path],
) -> Callable[..., Settings]:
    def factory(**overrides: Any) -> Settings:
        root = tmp_path / f"app-{len(list(tmp_path.iterdir()))}"
        root.mkdir(parents=True)
        data_path = write_knowledge_base(root / "data" / "knowledge.jsonl", sample_documents)
        evaluation_path = root / "data" / "golden.jsonl"
        evaluation_path.write_text(
            json.dumps(
                {
                    "id": "smoke-answer",
                    "query": "PyTorch 2.6 torch.load weights_only default error",
                    "project": "pytorch",
                    "version": "2.6",
                    "answerable": True,
                    "relevant_ids": ["pytorch-load-26"],
                    "tags": ["smoke"],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "smoke-refuse",
                    "query": "怎样制作酸面包面团",
                    "project": None,
                    "version": None,
                    "answerable": False,
                    "relevant_ids": [],
                    "expected_reason": "NO_EVIDENCE",
                    "tags": ["smoke"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        static_dir = root / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<h1>DevAssist test</h1>", encoding="utf-8")
        values: dict[str, Any] = {
            "root": root,
            "data_path": data_path,
            "evaluation_path": evaluation_path,
            "feedback_path": root / "runtime" / "feedback.jsonl",
            "reranker_weights_path": root / "artifacts" / "reranker" / "weights.json",
            "static_dir": static_dir,
            "top_k": 5,
            "candidate_k": 10,
            "confidence_threshold": 0.20,
            "max_query_chars": 12_000,
            "rate_limit_requests": 60,
            "rate_limit_window_seconds": 60,
            "admin_token": None,
        }
        values.update(overrides)
        return Settings(**values)

    return factory
