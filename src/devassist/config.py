"""Application configuration with explicit, testable environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root for both editable and source-tree execution."""

    configured = os.getenv("DEVASSIST_ROOT")
    if configured:
        return Path(configured).resolve()
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").exists():
        return source_candidate
    working_candidate = Path.cwd().resolve()
    if (working_candidate / "data" / "sample" / "knowledge_base.jsonl").exists():
        return working_candidate
    return source_candidate


def _env_int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _path_from_env(name: str, fallback: str, root: Path) -> Path:
    candidate = Path(os.getenv(name, fallback))
    return candidate if candidate.is_absolute() else root / candidate


def _secret_from_env(name: str, minimum_length: int = 16) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    if len(value) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} characters")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    The offline demo intentionally has no model API key and no external service
    requirement. Paths can be replaced in tests without global monkeypatching.
    """

    root: Path
    data_path: Path
    evaluation_path: Path
    feedback_path: Path
    reranker_weights_path: Path
    static_dir: Path
    top_k: int = 5
    candidate_k: int = 20
    confidence_threshold: float = 0.36
    max_query_chars: int = 12_000
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    admin_token: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: int = 20

    @classmethod
    def from_env(cls, root: Path | None = None) -> Settings:
        base = (root or repository_root()).resolve()
        static_dir = (
            base / "src" / "devassist" / "static"
            if root is not None
            else Path(__file__).resolve().parent / "static"
        )
        return cls(
            root=base,
            data_path=_path_from_env(
                "DEVASSIST_DATA_PATH", "data/sample/knowledge_base.jsonl", base
            ),
            evaluation_path=_path_from_env(
                "DEVASSIST_EVAL_PATH", "data/evaluation/golden_queries.jsonl", base
            ),
            feedback_path=_path_from_env(
                "DEVASSIST_FEEDBACK_PATH", "data/runtime/feedback.jsonl", base
            ),
            reranker_weights_path=_path_from_env(
                "DEVASSIST_RERANKER_WEIGHTS_PATH", "artifacts/reranker/weights.json", base
            ),
            static_dir=static_dir,
            top_k=_env_int("DEVASSIST_TOP_K", 5, maximum=8),
            candidate_k=_env_int("DEVASSIST_CANDIDATE_K", 20),
            confidence_threshold=_env_float("DEVASSIST_CONFIDENCE_THRESHOLD", 0.36, 0.0, 1.0),
            max_query_chars=_env_int("DEVASSIST_MAX_QUERY_CHARS", 12_000, 100, maximum=12_000),
            rate_limit_requests=_env_int("DEVASSIST_RATE_LIMIT_REQUESTS", 60),
            rate_limit_window_seconds=_env_int("DEVASSIST_RATE_LIMIT_WINDOW_SECONDS", 60),
            admin_token=_secret_from_env("DEVASSIST_ADMIN_TOKEN"),
            llm_base_url=os.getenv("DEVASSIST_LLM_BASE_URL") or None,
            llm_api_key=os.getenv("DEVASSIST_LLM_API_KEY") or None,
            llm_model=os.getenv("DEVASSIST_LLM_MODEL") or None,
            llm_timeout_seconds=_env_int("DEVASSIST_LLM_TIMEOUT_SECONDS", 20),
        )
