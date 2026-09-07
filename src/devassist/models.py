"""Validated public API contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=3, max_length=12_000)
]
ReasonCode = Literal[
    "ANSWERED", "NO_EVIDENCE", "LOW_CONFIDENCE", "VERSION_MISMATCH", "UNSAFE_QUERY"
]


class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str | None = Field(default=None, max_length=50, examples=["3.11.9"])
    cuda_version: str | None = Field(default=None, max_length=50, examples=["12.1"])
    operating_system: str | None = Field(default=None, max_length=120, examples=["Windows 11"])
    packages: dict[str, str] = Field(
        default_factory=dict, max_length=50, examples=[{"torch": "2.6.0"}]
    )

    @field_validator("packages")
    @classmethod
    def normalise_packages(cls, value: dict[str, str]) -> dict[str, str]:
        normalised: dict[str, str] = {}
        for key, version in value.items():
            package = str(key).strip().lower()
            package_version = str(version).strip()
            if not package or not package_version:
                raise ValueError("package names and versions must not be empty")
            if len(package) > 80 or len(package_version) > 80:
                raise ValueError("package names and versions must be at most 80 characters")
            normalised[package] = package_version
        return normalised


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText = Field(
        examples=["PyTorch 2.6 中 torch.load 为什么提示 weights_only 反序列化失败？"]
    )
    project: str | None = Field(default=None, max_length=50, examples=["pytorch"])
    version: str | None = Field(default=None, max_length=30, examples=["2.6"])
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    top_k: int = Field(default=5, ge=1, le=8)

    @field_validator("project", "version")
    @classmethod
    def normalise_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip().lower()
        return stripped or None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText
    project: str | None = Field(default=None, max_length=50)
    version: str | None = Field(default=None, max_length=30)
    source_types: list[Literal["documentation", "issue", "changelog", "migration"]] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"

    @field_validator("project", "version")
    @classmethod
    def normalise_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip().lower()
        return stripped or None


class SearchHit(BaseModel):
    document_id: str
    project: str
    version: str
    source_type: str
    title: str
    url: str
    excerpt: str
    score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float]
    reasons: list[str]


class Citation(BaseModel):
    document_id: str
    title: str
    url: str
    version: str
    source_type: str
    evidence: str


class DiagnoseResponse(BaseModel):
    request_id: str
    status: Literal["answered", "refused"]
    reason_code: ReasonCode
    diagnosis: str
    steps: list[str]
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: list[str]
    tools_used: list[str]
    safety_warnings: list[str]
    latency_ms: float = Field(ge=0.0)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1_000)


class FeedbackResponse(BaseModel):
    accepted: bool
    feedback_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    documents: int
    index_generation: int
    version: str


class MetricsResponse(BaseModel):
    requests: int
    answers: int
    refusals: int
    feedback_items: int
    p50_latency_ms: float
    p95_latency_ms: float
