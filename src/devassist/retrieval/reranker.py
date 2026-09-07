"""Explainable local feature reranker with optional learned weights."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from devassist.domain import KnowledgeDocument, RetrievalResult
from devassist.retrieval.text import lexical_overlap, tokenise, version_compatibility

DEFAULT_WEIGHTS: dict[str, float] = {
    "fused": 0.34,
    "body_overlap": 0.20,
    "title_overlap": 0.12,
    "code_signal": 0.13,
    "version": 0.12,
    "authority": 0.09,
}


class FeatureReranker:
    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights_path and weights_path.exists():
            loaded = json.loads(weights_path.read_text(encoding="utf-8"))
            for name in self.weights:
                if name in loaded:
                    self.weights[name] = float(loaded[name])

    def features(
        self,
        query: str,
        document: KnowledgeDocument,
        fused_score: float,
        requested_version: str | None,
    ) -> dict[str, float]:
        document_tokens = set(tokenise(document.searchable_text))
        code_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)?", query)
            if "_" in token or "." in token or token.endswith(("Error", "Exception", "Warning"))
        }
        code_hits = len(code_tokens & document_tokens) / max(len(code_tokens), 1)
        return {
            "fused": max(0.0, min(fused_score, 1.0)),
            "body_overlap": lexical_overlap(query, document.searchable_text),
            "title_overlap": lexical_overlap(query, document.title),
            "code_signal": code_hits,
            "version": version_compatibility(requested_version, document.version),
            "authority": document.authority,
        }

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        requested_version: str | None,
    ) -> list[RetrievalResult]:
        reranked: list[RetrievalResult] = []
        for result in results:
            features = self.features(query, result.document, result.score, requested_version)
            linear = sum(self.weights[name] * value for name, value in features.items())
            score = 1.0 / (1.0 + math.exp(-6.0 * (linear - 0.38)))
            reasons = list(result.reasons)
            if features["code_signal"] > 0:
                reasons.append("exact-code-or-error-token")
            if features["version"] >= 0.99 and requested_version:
                reasons.append("version-exact")
            components = dict(result.component_scores)
            components.update({f"feature_{name}": value for name, value in features.items()})
            reranked.append(
                RetrievalResult(
                    document=result.document,
                    score=max(0.0, min(float(score), 1.0)),
                    component_scores=components,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        reranked.sort(key=lambda item: (-item.score, item.document.id))
        return reranked
