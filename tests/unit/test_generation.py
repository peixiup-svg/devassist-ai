from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

from devassist.domain import KnowledgeDocument, RetrievalResult
from devassist.generation import (
    FallbackGenerator,
    OpenAICompatibleGenerator,
)


class FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def result(document: KnowledgeDocument) -> RetrievalResult:
    return RetrievalResult(document=document, score=0.9)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1/",
        "http://[::1]:8000/v1",
        "https://llm.example.test/v1",
    ],
)
def test_generator_accepts_https_and_loopback_http(base_url: str) -> None:
    generator = OpenAICompatibleGenerator(base_url=base_url, model="local-model")

    assert generator.endpoint.endswith("/chat/completions")
    assert "//chat/completions" not in generator.endpoint


@pytest.mark.parametrize(
    "base_url",
    [
        "http://llm.example.test/v1",
        "http://127.0.0.1.evil.test/v1",
        "http://localhost.evil.test/v1",
        "http://localhost@evil.test/v1",
        "ftp://localhost/v1",
        "llm.example.test/v1",
        "https:///missing-host",
    ],
)
def test_generator_rejects_unsafe_or_malformed_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS or loopback HTTP"):
        OpenAICompatibleGenerator(base_url=base_url, model="model")


def test_generator_rejects_blank_model() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        OpenAICompatibleGenerator(base_url="http://localhost:8000/v1", model="  ")


def test_generator_sends_bounded_grounded_payload_and_parses_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    captured: dict[str, Any] = {}
    content = {
        "choices": [
            {
                "message": {
                    "content": "```json\n"
                    + json.dumps(
                        {
                            "diagnosis": "  grounded diagnosis  ",
                            "steps": [" step one ", "step one", "", "step two"],
                        }
                    )
                    + "\n```"
                }
            }
        ]
    }

    def fake_urlopen(request: urllib.request.Request, *, timeout: int) -> FakeHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse(json.dumps(content).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    documents = tuple(document_factory(id=f"doc-{index}") for index in range(5))
    generator = OpenAICompatibleGenerator(
        base_url="http://127.0.0.1:8000/v1/",
        model="local-model",
        api_key="test-key",
        timeout_seconds=7,
    )

    diagnosis, steps = generator.generate(
        [result(document) for document in documents], query="Why does torch.load fail?"
    )

    assert diagnosis == "grounded diagnosis"
    assert steps == ["step one", "step two"]
    assert captured["timeout"] == 7
    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-key"
    payload = json.loads(request.data.decode())
    assert payload["model"] == "local-model"
    assert payload["temperature"] == 0
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload["question"] == "Why does torch.load fail?"
    assert [item["id"] for item in user_payload["evidence"]] == [
        "doc-0",
        "doc-1",
        "doc-2",
    ]
    assert "url" not in user_payload["evidence"][0]


def test_generator_without_api_key_omits_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    captured: list[urllib.request.Request] = []
    body = {"choices": [{"message": {"content": '{"diagnosis":"ok","steps":["one"]}'}}]}

    def fake_urlopen(request: urllib.request.Request, **_kwargs: object) -> FakeHTTPResponse:
        captured.append(request)
        return FakeHTTPResponse(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator(base_url="https://llm.example.test", model="model")

    assert generator.generate([result(document_factory())], query=None) == ("ok", ["one"])
    assert captured[0].get_header("Authorization") is None


def test_generator_empty_results_do_not_call_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("transport must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    generator = OpenAICompatibleGenerator(base_url="http://localhost:8000", model="model")

    assert generator.generate([], query="question") == ("", [])


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("down"),
        TimeoutError("timed out"),
        OSError("socket failed"),
    ],
)
def test_generator_wraps_transport_errors(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    def failing(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    generator = OpenAICompatibleGenerator(base_url="http://localhost:8000", model="model")

    with pytest.raises(RuntimeError, match="configured LLM generator failed"):
        generator.generate([result(document_factory())], query="query")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not json", "configured LLM generator failed"),
        (json.dumps({"choices": []}).encode(), "invalid structured response"),
        (
            json.dumps({"choices": [{"message": {"content": "not json"}}]}).encode(),
            "invalid structured response",
        ),
        (
            json.dumps(
                {"choices": [{"message": {"content": '{"diagnosis":"ok","steps":"bad"}'}}]}
            ).encode(),
            "invalid structured response",
        ),
        (
            json.dumps(
                {"choices": [{"message": {"content": '{"diagnosis":"ok","steps":[1]}'}}]}
            ).encode(),
            "invalid structured response",
        ),
        (
            json.dumps(
                {"choices": [{"message": {"content": '{"diagnosis":"","steps":["one"]}'}}]}
            ).encode(),
            "empty structured response",
        ),
        (
            json.dumps(
                {"choices": [{"message": {"content": '{"diagnosis":"ok","steps":[]}'}}]}
            ).encode(),
            "empty structured response",
        ),
    ],
)
def test_generator_fails_closed_on_malformed_responses(
    body: bytes,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse(body),
    )
    generator = OpenAICompatibleGenerator(base_url="http://localhost:8000", model="model")

    with pytest.raises(RuntimeError, match=message):
        generator.generate([result(document_factory())], query="query")


def test_generator_bounds_diagnosis_step_lengths_and_count(
    monkeypatch: pytest.MonkeyPatch,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"diagnosis": "d" * 1_500, "steps": [str(i) * 600 for i in range(8)]}
                    )
                }
            }
        ]
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse(json.dumps(response).encode()),
    )
    generator = OpenAICompatibleGenerator(base_url="http://localhost:8000", model="model")

    diagnosis, steps = generator.generate([result(document_factory())])

    assert len(diagnosis) == 1_200
    assert len(steps) == 6
    assert all(len(step) <= 500 for step in steps)


class RecordingGenerator:
    def __init__(self, outcome: tuple[str, list[str]] | Exception) -> None:
        self.outcome = outcome
        self.queries: list[str | None] = []

    def generate(
        self, results: list[RetrievalResult], *, query: str | None = None
    ) -> tuple[str, list[str]]:
        self.queries.append(query)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_fallback_generator_uses_fallback_only_for_runtime_failure(
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    primary = RecordingGenerator(RuntimeError("down"))
    fallback = RecordingGenerator(("fallback diagnosis", ["fallback step"]))
    generator = FallbackGenerator(primary, fallback)
    results = [result(document_factory())]

    assert generator.generate(results, query="original query") == (
        "fallback diagnosis",
        ["fallback step"],
    )
    assert primary.queries == ["original query"]
    assert fallback.queries == ["original query"]


def test_fallback_generator_returns_primary_and_does_not_hide_programming_errors(
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    results = [result(document_factory())]
    primary = RecordingGenerator(("primary", ["step"]))
    fallback = RecordingGenerator(("fallback", ["step"]))

    assert FallbackGenerator(primary, fallback).generate(results, query="q") == (
        "primary",
        ["step"],
    )
    assert fallback.queries == []

    programming_error = RecordingGenerator(ValueError("bug"))
    with pytest.raises(ValueError, match="bug"):
        FallbackGenerator(programming_error, fallback).generate(results, query="q")


def test_fallback_defaults_to_extractive_generator(
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    document = document_factory()
    generator = FallbackGenerator(RecordingGenerator(RuntimeError("offline")))

    assert generator.generate([result(document)], query="q") == (
        document.summary,
        list(document.resolution),
    )
