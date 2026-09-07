"""Bounded retrieve-generate-validate workflow."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from devassist.analyzer import QueryAnalyzer
from devassist.citations import CitationValidator
from devassist.domain import RetrievalResult
from devassist.generation import EVIDENCE_LIMIT, ExtractiveGenerator, Generator
from devassist.models import DiagnoseRequest, DiagnoseResponse, ReasonCode
from devassist.retrieval.hybrid import HybridRetriever
from devassist.retrieval.text import version_is_compatible, version_parts


class DevAssistAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        confidence_threshold: float = 0.36,
        analyzer: QueryAnalyzer | None = None,
        generator: Generator | None = None,
        citation_validator: CitationValidator | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.analyzer = analyzer or QueryAnalyzer()
        self.generator = generator or ExtractiveGenerator()
        self.citation_validator = citation_validator or CitationValidator()
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.clock = clock or time.perf_counter

    def diagnose(self, request: DiagnoseRequest) -> DiagnoseResponse:
        started = self.clock()
        request_id = self.id_factory()
        analysis = self.analyzer.analyse(
            request.query,
            project=request.project,
            version=request.version,
            environment=request.environment,
        )
        source_filter = ",".join(analysis.source_types)
        tools = [f"hybrid_retrieval[source_types={source_filter}]"]

        def latency() -> float:
            return round((self.clock() - started) * 1000.0, 3)

        if self._is_only_unsafe_instruction(analysis.clean_query, analysis.safety_warnings):
            return self._refusal(
                request_id,
                "UNSAFE_QUERY",
                "请求主要包含试图改变系统行为的指令，未发现可验证的技术问题。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        if analysis.version and analysis.version.lower() in {"latest", "any", "all"}:
            return self._refusal(
                request_id,
                "VERSION_MISMATCH",
                "开放版本标识无法验证，请提供明确的数字版本（例如 2.6.0）。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        if self._version_outside_known_coverage(analysis.project, analysis.version):
            return self._refusal(
                request_id,
                "VERSION_MISMATCH",
                f"请求版本 {analysis.version} 超出当前知识库已验证的版本范围。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        results = self.retriever.search(
            analysis.clean_query,
            project=analysis.project,
            version=analysis.version,
            source_types=analysis.source_types,
            top_k=request.top_k,
            mode="hybrid_rerank",
        )
        if not results:
            return self._refusal(
                request_id,
                "NO_EVIDENCE",
                "当前知识库中没有找到可验证的相关资料。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        if not self._specific_error_supported(analysis.clean_query, results):
            return self._refusal(
                request_id,
                "NO_EVIDENCE",
                "检索资料没有覆盖问题中的关键错误特征，不能用同框架的其他主题替代回答。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        if analysis.version and not any(
            version_is_compatible(analysis.version, result.document.version) for result in results
        ):
            return self._refusal(
                request_id,
                "VERSION_MISMATCH",
                f"找到了相似资料，但没有与请求版本 {analysis.version} 兼容的证据。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        confidence, has_evidence = self._confidence(results)
        if not has_evidence:
            return self._refusal(
                request_id,
                "NO_EVIDENCE",
                "检索结果只有表面相似词，无法支撑可靠诊断。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )
        if confidence < self.confidence_threshold:
            return self._refusal(
                request_id,
                "LOW_CONFIDENCE",
                "资料相关性不足，继续生成可能造成误导。请补充完整报错和版本信息。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        diagnosis, steps = self.generator.generate(results, query=analysis.clean_query)
        citations = self.citation_validator.build(results, limit=EVIDENCE_LIMIT)
        if not steps or not self.citation_validator.validate(citations, results):
            return self._refusal(
                request_id,
                "NO_EVIDENCE",
                "资料无法形成带有有效引用的诊断。",
                analysis.missing_information,
                tools,
                analysis.safety_warnings,
                latency(),
            )

        return DiagnoseResponse(
            request_id=request_id,
            status="answered",
            reason_code="ANSWERED",
            diagnosis=diagnosis,
            steps=steps,
            citations=citations,
            confidence=confidence,
            missing_information=list(analysis.missing_information),
            tools_used=tools,
            safety_warnings=list(analysis.safety_warnings),
            latency_ms=latency(),
        )

    @staticmethod
    def _confidence(results: list[RetrievalResult]) -> tuple[float, bool]:
        top = results[0]
        components = top.component_scores
        bm25 = min(components.get("bm25_raw", 0.0) / 4.0, 1.0)
        dense = components.get("dense_cosine", 0.0)
        overlap = components.get("feature_body_overlap", 0.0)
        code_signal = components.get("feature_code_signal", 0.0)
        version = components.get("version_compatibility", 1.0)
        raw_signal = max(bm25, dense, overlap, code_signal)
        # LSA is a broad candidate generator, not standalone proof. Grounding
        # requires a lexical/code anchor in the curated evidence.
        has_anchor = overlap >= 0.035 or code_signal > 0
        has_evidence = (
            raw_signal >= 0.035
            and has_anchor
            and (components.get("bm25_raw", 0.0) > 0.15 or dense > 0.045 or code_signal > 0)
        )
        confidence = (
            0.40 * top.score
            + 0.22 * bm25
            + 0.14 * dense
            + 0.09 * overlap
            + 0.07 * code_signal
            + 0.05 * version
            + 0.03 * top.document.authority
        )
        return round(max(0.0, min(confidence, 0.99)), 4), has_evidence

    def _version_outside_known_coverage(
        self, project: str | None, requested_version: str | None
    ) -> bool:
        wanted = version_parts(requested_version)
        if not project or wanted is None or len(wanted) < 2:
            return False
        known = [
            parts
            for document in self.retriever.documents
            if document.project == project
            if (parts := version_parts(document.version)) is not None
        ]
        known_major_minor = [(parts[0], parts[1]) for parts in known if len(parts) >= 2]
        if not known_major_minor:
            return False
        requested_major_minor = (wanted[0], wanted[1])
        return not min(known_major_minor) <= requested_major_minor <= max(known_major_minor)

    @staticmethod
    def _specific_error_supported(query: str, results: list[RetrievalResult]) -> bool:
        """Require topic-level evidence for distinctive failure signatures."""

        signature_groups = (("out of memory", "cuda oom", "显存溢出", "内存不足"),)
        lowered_query = query.lower()
        evidence = " ".join(result.document.searchable_text.lower() for result in results[:3])
        for signatures in signature_groups:
            if any(signature in lowered_query for signature in signatures):
                return any(signature in evidence for signature in signatures)
        return True

    @staticmethod
    def _is_only_unsafe_instruction(query: str, warnings: tuple[str, ...]) -> bool:
        if not warnings:
            return False
        technical_markers = (
            "python",
            "torch",
            "pytorch",
            "fastapi",
            "transformers",
            "cuda",
            "error",
            "exception",
            "报错",
        )
        lowered = query.lower()
        return not any(marker in lowered for marker in technical_markers)

    @staticmethod
    def _refusal(
        request_id: str,
        reason_code: ReasonCode,
        diagnosis: str,
        missing_information: tuple[str, ...],
        tools: list[str],
        safety_warnings: tuple[str, ...],
        latency_ms: float,
    ) -> DiagnoseResponse:
        return DiagnoseResponse(
            request_id=request_id,
            status="refused",
            reason_code=reason_code,
            diagnosis=diagnosis,
            steps=[],
            citations=[],
            confidence=0.0,
            missing_information=list(missing_information),
            tools_used=tools,
            safety_warnings=list(safety_warnings),
            latency_ms=latency_ms,
        )
