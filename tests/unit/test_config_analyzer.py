from __future__ import annotations

import socket
from pathlib import Path

import pytest

from devassist.analyzer import QueryAnalyzer
from devassist.config import Settings, repository_root
from devassist.models import EnvironmentInfo


def test_settings_loads_relative_paths_and_runtime_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEVASSIST_DATA_PATH", "fixtures/kb.jsonl")
    monkeypatch.setenv("DEVASSIST_EVAL_PATH", str(tmp_path / "absolute-eval.jsonl"))
    monkeypatch.setenv("DEVASSIST_FEEDBACK_PATH", "runtime/feedback.jsonl")
    monkeypatch.setenv("DEVASSIST_RERANKER_WEIGHTS_PATH", "weights/reranker.json")
    monkeypatch.setenv("DEVASSIST_TOP_K", "7")
    monkeypatch.setenv("DEVASSIST_CANDIDATE_K", "30")
    monkeypatch.setenv("DEVASSIST_CONFIDENCE_THRESHOLD", "0.42")
    monkeypatch.setenv("DEVASSIST_MAX_QUERY_CHARS", "500")
    monkeypatch.setenv("DEVASSIST_RATE_LIMIT_REQUESTS", "12")
    monkeypatch.setenv("DEVASSIST_RATE_LIMIT_WINDOW_SECONDS", "9")
    monkeypatch.setenv("DEVASSIST_ADMIN_TOKEN", "admin-token-12345")
    monkeypatch.setenv("DEVASSIST_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("DEVASSIST_LLM_API_KEY", "secret")
    monkeypatch.setenv("DEVASSIST_LLM_MODEL", "local")
    monkeypatch.setenv("DEVASSIST_LLM_TIMEOUT_SECONDS", "4")

    settings = Settings.from_env(tmp_path)

    assert settings.root == tmp_path.resolve()
    assert settings.data_path == tmp_path / "fixtures" / "kb.jsonl"
    assert settings.evaluation_path == tmp_path / "absolute-eval.jsonl"
    assert settings.feedback_path == tmp_path / "runtime" / "feedback.jsonl"
    assert settings.reranker_weights_path == tmp_path / "weights" / "reranker.json"
    assert settings.static_dir == tmp_path / "src" / "devassist" / "static"
    assert settings.top_k == 7
    assert settings.candidate_k == 30
    assert settings.confidence_threshold == 0.42
    assert settings.max_query_chars == 500
    assert settings.rate_limit_requests == 12
    assert settings.rate_limit_window_seconds == 9
    assert settings.admin_token == "admin-token-12345"
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_api_key == "secret"
    assert settings.llm_model == "local"
    assert settings.llm_timeout_seconds == 4


def test_settings_defaults_and_repository_root_are_resolvable(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)

    assert settings.data_path == tmp_path / "data" / "sample" / "knowledge_base.jsonl"
    assert settings.evaluation_path == tmp_path / "data" / "evaluation" / "golden_queries.jsonl"
    assert settings.top_k == 5
    assert settings.llm_base_url is None
    assert repository_root().name == "devassist-ai"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("DEVASSIST_TOP_K", "0", "must be >= 1"),
        ("DEVASSIST_TOP_K", "9", "must be <= 8"),
        ("DEVASSIST_MAX_QUERY_CHARS", "99", "must be >= 100"),
        ("DEVASSIST_MAX_QUERY_CHARS", "12001", "must be <= 12000"),
        ("DEVASSIST_LLM_TIMEOUT_SECONDS", "not-an-int", "invalid literal"),
        ("DEVASSIST_CONFIDENCE_THRESHOLD", "-0.1", "between 0.0 and 1.0"),
        ("DEVASSIST_CONFIDENCE_THRESHOLD", "1.1", "between 0.0 and 1.0"),
        ("DEVASSIST_ADMIN_TOKEN", "too-short", "at least 16 characters"),
    ],
)
def test_settings_rejects_invalid_environment_values(
    name: str,
    value: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env(tmp_path)


def test_analyzer_detects_project_version_intent_and_missing_context() -> None:
    analysis = QueryAnalyzer().analyse(
        "CUDA RuntimeError: device-side assert triggered",
        project=None,
        version=None,
        environment=EnvironmentInfo(packages={"Torch": ">=2.6.0"}),
    )

    assert analysis.project == "pytorch"
    assert analysis.version == "2.6.0"
    assert analysis.intent == "diagnosis"
    assert analysis.exception_names == ("RuntimeError",)
    assert "documentation" in analysis.source_types
    assert "issue" in analysis.source_types
    assert "Python 版本" in analysis.missing_information
    assert "CUDA 版本" in analysis.missing_information
    assert "操作系统" in analysis.missing_information


@pytest.mark.parametrize(
    ("query", "project", "intent"),
    [
        ("Transformers 4.46 migration changelog", "transformers", "migration"),
        ("FastAPI 0.115 version compatibility", "fastapi", "compatibility"),
        ("PyTorch inference_mode documentation", "pytorch", "documentation"),
    ],
)
def test_analyzer_routes_supported_intents(query: str, project: str, intent: str) -> None:
    analysis = QueryAnalyzer().analyse(
        query,
        project=None,
        version=None,
        environment=EnvironmentInfo(),
    )

    assert analysis.project == project
    assert analysis.intent == intent


def test_analyzer_prefers_explicit_version_then_environment_then_query() -> None:
    analyzer = QueryAnalyzer()
    environment = EnvironmentInfo(packages={"torch": "~=2.5.1"})

    explicit = analyzer.analyse(
        "PyTorch v2.4 documentation",
        project="PyTorch",
        version="2.6",
        environment=environment,
    )
    from_environment = analyzer.analyse(
        "PyTorch v2.4 documentation",
        project="pytorch",
        version=None,
        environment=environment,
    )
    from_query = analyzer.analyse(
        "PyTorch v2.4 documentation",
        project="pytorch",
        version=None,
        environment=EnvironmentInfo(),
    )

    assert explicit.version == "2.6"
    assert from_environment.version == "2.5.1"
    assert from_query.version == "2.4"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Python 3.11 with PyTorch 2.6 torch.load error", "2.6"),
        ("CUDA 12.1 + PyTorch 2.6 RuntimeError", "2.6"),
        ("Pydantic 2.10 with FastAPI 0.115 validation error", "0.115"),
        ("Uvicorn 0.34 with FastAPI 0.115 startup error", "0.115"),
        ("PyTorch 2.6 on Python 3.11 and CUDA 12.1", "2.6"),
        ("PyTorch on Python 3.11 has an error", None),
    ],
)
def test_analyzer_associates_versions_with_the_detected_project(
    query: str, expected: str | None
) -> None:
    analysis = QueryAnalyzer().analyse(
        query,
        project=None,
        version=None,
        environment=EnvironmentInfo(),
    )

    expected_project = "fastapi" if "FastAPI" in query else "pytorch"
    assert analysis.project == expected_project
    assert analysis.version == expected


def test_analyzer_does_not_treat_dependency_version_as_fastapi_version() -> None:
    analysis = QueryAnalyzer().analyse(
        "FastAPI validation error",
        project=None,
        version=None,
        environment=EnvironmentInfo(packages={"pydantic": "2.10.6"}),
    )

    assert analysis.project == "fastapi"
    assert analysis.version is None


def test_analyzer_prefers_explicit_framework_name_over_backend_hints() -> None:
    analysis = QueryAnalyzer().analyse(
        "Transformers 4.48 Trainer with torch 2.5 CUDA out of memory",
        project=None,
        version=None,
        environment=EnvironmentInfo(),
    )

    assert analysis.project == "transformers"
    assert analysis.version == "4.48"


def test_analyzer_canonicalises_structured_project_alias() -> None:
    analysis = QueryAnalyzer().analyse(
        "torch.load weights_only error",
        project=" Torch ",
        version="2.6",
        environment=EnvironmentInfo(),
    )

    assert analysis.project == "pytorch"


def test_analyzer_cleans_secrets_and_preserves_safety_warning() -> None:
    analysis = QueryAnalyzer().analyse(
        "Python error api_key=abcdefgh12345678",
        project=None,
        version=None,
        environment=EnvironmentInfo(),
    )

    assert "abcdefgh12345678" not in analysis.clean_query
    assert "[REDACTED_SECRET]" in analysis.clean_query
    assert analysis.safety_warnings


def test_test_suite_blocks_external_socket_connections() -> None:
    with socket.socket() as client, pytest.raises(AssertionError, match="external network"):
        client.connect(("203.0.113.1", 80))
