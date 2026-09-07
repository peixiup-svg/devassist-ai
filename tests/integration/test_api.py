from __future__ import annotations

import importlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from devassist.api.app import FixedWindowRateLimiter, create_app
from devassist.config import Settings


@pytest.fixture
def api_client(
    settings_factory: Callable[..., Settings],
) -> Iterator[tuple[TestClient, Settings]]:
    settings = settings_factory()
    with TestClient(create_app(settings)) as client:
        yield client, settings


def test_home_and_health_expose_ready_offline_service(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client

    home = client.get("/")
    health = client.get("/health")

    assert home.status_code == 200
    assert "DevAssist test" in home.text
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["documents"] == 4
    assert health.json()["index_generation"] == 1
    assert health.json()["version"]


def test_search_returns_version_filtered_explainable_hits(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client
    payload = {
        "query": "torch.load weights_only default",
        "project": "PyTorch",
        "version": "2.6",
        "source_types": ["changelog"],
        "top_k": 3,
        "mode": "hybrid_rerank",
    }

    response = client.post("/v1/search", json=payload)

    assert response.status_code == 200
    hits = response.json()
    assert [hit["document_id"] for hit in hits] == ["pytorch-load-26"]
    assert hits[0]["version"] == "2.6"
    assert hits[0]["source_type"] == "changelog"
    assert hits[0]["url"].startswith("https://")
    assert 0.0 <= hits[0]["score"] <= 1.0
    assert hits[0]["component_scores"]
    assert hits[0]["reasons"]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_search_supports_each_documented_mode(
    mode: str,
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client

    response = client.post(
        "/v1/search",
        json={"query": "tokenizer padding token", "project": "transformers", "mode": mode},
    )

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "transformers-padding"


def test_diagnose_answers_with_citations_and_updates_metrics(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client

    response = client.post(
        "/v1/diagnose",
        json={
            "query": "PyTorch 2.6 torch.load weights_only UnpicklingError 怎么修复",
            "project": "pytorch",
            "version": "2.6",
            "environment": {"python_version": "3.11", "packages": {"torch": "2.6"}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["reason_code"] == "ANSWERED"
    assert body["steps"]
    assert body["citations"]
    assert body["citations"][0]["document_id"] == "pytorch-load-26"
    assert body["tools_used"] == [
        "hybrid_retrieval[source_types=documentation,issue,changelog,migration]"
    ]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0

    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["requests"] == 1
    assert metrics.json()["answers"] == 1
    assert metrics.json()["refusals"] == 0


def test_diagnose_refuses_out_of_domain_and_future_version(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client

    out_of_domain = client.post("/v1/diagnose", json={"query": "怎样制作酸面包面团"})
    future = client.post(
        "/v1/diagnose",
        json={
            "query": "PyTorch 99.0 torch.load weights_only 有什么变化",
            "project": "pytorch",
            "version": "99.0",
        },
    )

    assert out_of_domain.status_code == 200
    assert out_of_domain.json()["status"] == "refused"
    assert out_of_domain.json()["reason_code"] == "NO_EVIDENCE"
    assert out_of_domain.json()["citations"] == []
    assert future.status_code == 200
    assert future.json()["reason_code"] == "VERSION_MISMATCH"
    assert client.get("/v1/metrics").json()["refusals"] == 2


def test_feedback_is_redacted_persisted_and_counted(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, settings = api_client
    secret = "sk-abcdefghijklmnop1234"

    response = client.post(
        "/v1/feedback",
        json={"request_id": "a" * 32, "rating": -1, "comment": f"泄漏 {secret}"},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert len(response.json()["feedback_id"]) == 32
    lines = settings.feedback_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["request_id"] == "a" * 32
    assert persisted["rating"] == -1
    assert persisted["comment"] == "泄漏 [REDACTED_SECRET]"
    assert secret not in lines[0]
    assert persisted["created_at"].endswith("+00:00")
    assert client.get("/v1/metrics").json()["feedback_items"] == 1


@pytest.mark.parametrize(
    ("path", "payload", "status_code"),
    [
        ("/v1/search", {"query": "ok"}, 422),
        ("/v1/search", {"query": "valid query", "mode": "magic"}, 422),
        ("/v1/search", {"query": "valid query", "top_k": 21}, 422),
        ("/v1/search", {"query": "valid query", "extra": True}, 422),
        ("/v1/diagnose", {"query": "valid query", "top_k": 9}, 422),
        ("/v1/feedback", {"request_id": "short", "rating": 1}, 422),
        ("/v1/feedback", {"request_id": "a" * 32, "rating": 0}, 422),
    ],
)
def test_api_rejects_invalid_contracts(
    path: str,
    payload: dict[str, Any],
    status_code: int,
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client

    response = client.post(path, json=payload)

    assert response.status_code == status_code
    assert response.json()["detail"]


@pytest.mark.parametrize("path", ["/v1/search", "/v1/diagnose"])
def test_api_applies_runtime_query_size_limit(
    path: str,
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory(max_query_chars=20)
    with TestClient(create_app(settings)) as client:
        response = client.post(path, json={"query": "x" * 21})

    assert response.status_code == 413
    assert response.json() == {"detail": "query is too large"}


def test_not_ready_service_fails_closed(
    api_client: tuple[TestClient, Settings],
) -> None:
    client, _settings = api_client
    client.app.state.service._agent = None

    health = client.get("/health")
    diagnose = client.post("/v1/diagnose", json={"query": "valid technical query"})

    assert health.status_code == 503
    assert diagnose.status_code == 503
    assert health.json() == {"detail": "index not ready"}


def test_admin_endpoint_can_be_disabled_or_protected(
    settings_factory: Callable[..., Settings],
) -> None:
    disabled = settings_factory(admin_token=None)
    with TestClient(create_app(disabled)) as client:
        assert client.post("/admin/reindex").status_code == 503

    protected = settings_factory(admin_token="correct-token-123")
    with TestClient(create_app(protected)) as client:
        assert client.post("/admin/reindex").status_code == 401
        assert (
            client.post("/admin/reindex", headers={"x-admin-token": "wrong-token"}).status_code
            == 401
        )
        response = client.post("/admin/reindex", headers={"x-admin-token": "correct-token-123"})

    assert response.status_code == 200
    assert response.json()["index_generation"] == 2
    assert response.json()["documents"] == 4


def test_rate_limit_ignores_spoofed_forwarded_headers(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory(rate_limit_requests=2, rate_limit_window_seconds=60)
    payload = {"query": "torch.load weights_only"}
    first_client = {"x-forwarded-for": "203.0.113.1, 10.0.0.1"}
    spoofed_other_client = {"x-forwarded-for": "203.0.113.2"}

    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/search", json=payload, headers=first_client).status_code == 200
        assert client.post("/v1/search", json=payload, headers=first_client).status_code == 200
        rejected = client.post("/v1/search", json=payload, headers=first_client)
        assert rejected.status_code == 429
        assert rejected.json() == {"detail": "rate limit exceeded"}
        assert (
            client.post("/v1/search", json=payload, headers=spoofed_other_client).status_code == 429
        )


def test_fixed_window_limiter_expires_old_events(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0.0, 0.5, 1.1))
    api_module = importlib.import_module("devassist.api.app")
    monkeypatch.setattr(api_module.time, "monotonic", lambda: next(ticks))
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=1)

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is True


def test_fixed_window_limiter_bounds_tracked_client_keys() -> None:
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=60, max_keys=2)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-c") is False


def test_service_uses_configured_llm_but_falls_back_offline(
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    settings = settings_factory(
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="local-model",
        llm_api_key="local-test-key",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/diagnose",
            json={
                "query": "PyTorch 2.6 torch.load weights_only UnpicklingError",
                "project": "pytorch",
                "version": "2.6",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["citations"][0]["document_id"] == "pytorch-load-26"


def test_unknown_route_returns_json_404(api_client: tuple[TestClient, Settings]) -> None:
    client, _settings = api_client

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
