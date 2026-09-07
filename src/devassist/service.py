"""Application service: atomic indexing, retrieval, diagnosis, and feedback."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from devassist.agent import DevAssistAgent
from devassist.config import Settings
from devassist.generation import (
    ExtractiveGenerator,
    FallbackGenerator,
    Generator,
    OpenAICompatibleGenerator,
)
from devassist.metrics import AppMetrics
from devassist.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    FeedbackRequest,
    FeedbackResponse,
    SearchHit,
    SearchRequest,
)
from devassist.retrieval.hybrid import HybridRetriever
from devassist.retrieval.reranker import FeatureReranker
from devassist.retrieval.text import compact_evidence
from devassist.security import safe_comment
from devassist.store import KnowledgeStore


class DevAssistService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.store = KnowledgeStore(self.settings.data_path)
        self.metrics = AppMetrics()
        self._feedback_lock = threading.Lock()
        self._index_lock = threading.RLock()
        self._reindex_lock = threading.Lock()
        self._retriever: HybridRetriever | None = None
        self._agent: DevAssistAgent | None = None
        self.reindex()

    @property
    def ready(self) -> bool:
        return self._retriever is not None and self._agent is not None

    @property
    def document_count(self) -> int:
        return len(self.store.documents)

    def reindex(self) -> int:
        """Build a complete new index before swapping it into service."""

        with self._reindex_lock:
            next_documents = self.store.read()
            next_retriever = HybridRetriever(
                next_documents,
                candidate_k=self.settings.candidate_k,
                reranker=FeatureReranker(self.settings.reranker_weights_path),
            )
            next_agent = DevAssistAgent(
                next_retriever,
                confidence_threshold=self.settings.confidence_threshold,
                generator=self._build_generator(),
            )
            with self._index_lock:
                generation = self.store.replace(next_documents)
                self._retriever = next_retriever
                self._agent = next_agent
            return generation

    def _build_generator(self) -> Generator:
        settings = self.settings
        if settings.llm_base_url and settings.llm_model:
            primary = OpenAICompatibleGenerator(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                timeout_seconds=settings.llm_timeout_seconds,
            )
            return FallbackGenerator(primary, ExtractiveGenerator())
        return ExtractiveGenerator()

    def search(self, request: SearchRequest) -> list[SearchHit]:
        with self._index_lock:
            retriever = self._retriever
        if retriever is None:
            raise RuntimeError("index is not ready")
        results = retriever.search(
            request.query,
            project=request.project,
            version=request.version,
            source_types=request.source_types,
            top_k=(request.top_k if "top_k" in request.model_fields_set else self.settings.top_k),
            mode=request.mode,
        )
        return [
            SearchHit(
                document_id=result.document.id,
                project=result.document.project,
                version=result.document.version,
                source_type=result.document.source_type,
                title=result.document.title,
                url=result.document.url,
                excerpt=compact_evidence(result.document.content, 240),
                score=round(result.score, 6),
                component_scores={
                    key: round(value, 6) for key, value in result.component_scores.items()
                },
                reasons=list(result.reasons),
            )
            for result in results
        ]

    def diagnose(self, request: DiagnoseRequest) -> DiagnoseResponse:
        with self._index_lock:
            agent = self._agent
        if agent is None:
            raise RuntimeError("index is not ready")
        if "top_k" not in request.model_fields_set:
            request = request.model_copy(update={"top_k": self.settings.top_k})
        response = agent.diagnose(request)
        self.metrics.record_diagnosis(
            answered=response.status == "answered", latency_ms=response.latency_ms
        )
        return response

    def save_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        feedback_id = uuid.uuid4().hex
        payload = {
            "feedback_id": feedback_id,
            "request_id": request.request_id,
            "rating": request.rating,
            "comment": safe_comment(request.comment),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path: Path = self.settings.feedback_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._feedback_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.metrics.record_feedback()
        return FeedbackResponse(accepted=True, feedback_id=feedback_id)
