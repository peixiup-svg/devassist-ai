"""BM25 + LSA retrieval with weighted reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from devassist.domain import KnowledgeDocument, RetrievalResult
from devassist.projects import canonical_project
from devassist.retrieval.bm25 import BM25Index
from devassist.retrieval.dense import LSADenseIndex
from devassist.retrieval.reranker import FeatureReranker
from devassist.retrieval.text import version_compatibility, version_is_compatible


class HybridRetriever:
    MODES: ClassVar[set[str]] = {"bm25", "dense", "hybrid", "hybrid_rerank"}

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        candidate_k: int = 20,
        rrf_constant: int = 60,
        reranker: FeatureReranker | None = None,
    ) -> None:
        if candidate_k <= 0 or rrf_constant <= 0:
            raise ValueError("candidate_k and rrf_constant must be positive")
        self.documents = tuple(documents)
        texts = [document.searchable_text for document in self.documents]
        self.bm25 = BM25Index(texts)
        self.dense = LSADenseIndex(texts)
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.reranker = reranker or FeatureReranker()

    def _candidates(
        self,
        project: str | None,
        source_types: Sequence[str] | None,
    ) -> list[int]:
        wanted_project = canonical_project(project)
        wanted_types = {item.lower() for item in source_types} if source_types else None
        return [
            index
            for index, document in enumerate(self.documents)
            if (wanted_project is None or document.project == wanted_project)
            and (wanted_types is None or document.source_type in wanted_types)
        ]

    @staticmethod
    def _normalise(raw: float, maximum: float) -> float:
        if maximum <= 0:
            return 0.0
        return max(0.0, min(raw / maximum, 1.0))

    def search(
        self,
        query: str,
        *,
        project: str | None = None,
        version: str | None = None,
        source_types: Sequence[str] | None = None,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
    ) -> list[RetrievalResult]:
        if mode not in self.MODES:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        if top_k <= 0 or not query.strip():
            return []
        candidates = self._candidates(project, source_types)
        if not candidates:
            return []
        candidate_k = min(max(self.candidate_k, top_k), len(candidates))
        # Backends can return deterministic zero-score rows to keep their low-level
        # matrix APIs total. They are not evidence and must never receive RRF credit.
        epsilon = 1e-12
        sparse = [
            item
            for item in self.bm25.search(query, candidate_indices=candidates, top_k=candidate_k)
            if item[1] > epsilon
        ]
        dense = [
            item
            for item in self.dense.search(query, candidate_indices=candidates, top_k=candidate_k)
            if item[1] > epsilon
        ]
        if not sparse and not dense:
            return []
        sparse_map = {index: (rank, score) for rank, (index, score) in enumerate(sparse, start=1)}
        dense_map = {index: (rank, score) for rank, (index, score) in enumerate(dense, start=1)}
        sparse_max = max((score for _, score in sparse), default=0.0)
        dense_max = max((score for _, score in dense), default=0.0)

        selected_indices: set[int]
        if mode == "bm25":
            selected_indices = set(sparse_map)
        elif mode == "dense":
            selected_indices = set(dense_map)
        else:
            selected_indices = set(sparse_map) | set(dense_map)

        results: list[RetrievalResult] = []
        ideal_rrf = 2.0 / (self.rrf_constant + 1)
        for index in selected_indices:
            sparse_rank, sparse_raw = sparse_map.get(index, (0, 0.0))
            dense_rank, dense_raw = dense_map.get(index, (0, 0.0))
            sparse_norm = self._normalise(sparse_raw, sparse_max)
            dense_norm = self._normalise(dense_raw, dense_max)
            if mode == "bm25":
                fused = sparse_norm
            elif mode == "dense":
                fused = dense_norm
            else:
                rrf = 0.0
                if sparse_rank:
                    rrf += 1.0 / (self.rrf_constant + sparse_rank)
                if dense_rank:
                    rrf += 1.0 / (self.rrf_constant + dense_rank)
                rrf_norm = rrf / ideal_rrf
                fused = 0.55 * rrf_norm + 0.25 * sparse_norm + 0.20 * dense_norm

            document = self.documents[index]
            version_score = version_compatibility(version, document.version)
            score = max(0.0, min(fused * (0.82 + 0.18 * version_score), 1.0))
            reasons: list[str] = []
            if sparse_rank and sparse_raw > 0:
                reasons.append(f"bm25-rank-{sparse_rank}")
            if dense_rank and dense_raw > 0:
                reasons.append(f"dense-rank-{dense_rank}")
            if version and version_score < 0.9:
                reasons.append("version-fallback")
            results.append(
                RetrievalResult(
                    document=document,
                    score=score,
                    component_scores={
                        "bm25_raw": float(sparse_raw),
                        "bm25_normalised": sparse_norm,
                        "dense_cosine": float(dense_raw),
                        "dense_normalised": dense_norm,
                        "version_compatibility": version_score,
                    },
                    reasons=tuple(reasons),
                )
            )

        results.sort(key=lambda item: (-item.score, item.document.id))
        if mode == "hybrid_rerank":
            results = self.reranker.rerank(query, results, requested_version=version)
        if version:
            compatible = [
                result
                for result in results
                if version_is_compatible(version, result.document.version)
            ]
            # Fail closed at the agent layer when no compatible evidence exists;
            # never mix incompatible version-specific documents into good evidence.
            if compatible:
                results = compatible
        return results[:top_k]
