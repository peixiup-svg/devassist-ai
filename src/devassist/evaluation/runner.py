"""Run reproducible ablations on the committed demo benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devassist.citations import CitationValidator
from devassist.config import Settings
from devassist.models import DiagnoseRequest, EnvironmentInfo
from devassist.retrieval.hybrid import HybridRetriever
from devassist.retrieval.reranker import FeatureReranker
from devassist.retrieval.text import version_is_compatible
from devassist.service import DevAssistService

from .metrics import binary_scores, ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    project: str | None
    version: str | None
    answerable: bool
    relevant_ids: frozenset[str]
    tags: tuple[str, ...]
    expected_reason: str | None = None


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation JSON at line {line_number}") from exc
            case = EvaluationCase(
                id=str(payload["id"]),
                query=str(payload["query"]),
                project=(str(payload["project"]).lower() if payload.get("project") else None),
                version=(str(payload["version"]).lower() if payload.get("version") else None),
                answerable=bool(payload["answerable"]),
                relevant_ids=frozenset(str(item) for item in payload.get("relevant_ids", [])),
                tags=tuple(str(item) for item in payload.get("tags", [])),
                expected_reason=(
                    str(payload["expected_reason"]) if payload.get("expected_reason") else None
                ),
            )
            if case.id in seen:
                raise ValueError(f"duplicate evaluation case id: {case.id}")
            if case.answerable != bool(case.relevant_ids):
                raise ValueError(f"case {case.id} has inconsistent answerable/relevant_ids")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation set is empty")
    return cases


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate(settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or Settings.from_env()
    started = time.perf_counter()
    service = DevAssistService(resolved)
    retriever = HybridRetriever(
        service.store.documents,
        candidate_k=resolved.candidate_k,
        reranker=FeatureReranker(resolved.reranker_weights_path),
    )
    cases = load_cases(resolved.evaluation_path)
    answerable = [case for case in cases if case.answerable]
    ablations: dict[str, dict[str, float | int]] = {}

    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        recall_values: list[float] = []
        reciprocal_values: list[float] = []
        ndcg_values: list[float] = []
        version_values: list[float] = []
        for case in answerable:
            results = retriever.search(
                case.query,
                project=case.project,
                version=case.version,
                top_k=10,
                mode=mode,
            )
            ranked_ids = [result.document.id for result in results]
            recall_values.append(recall_at_k(ranked_ids, set(case.relevant_ids), 5))
            reciprocal_values.append(reciprocal_rank(ranked_ids, set(case.relevant_ids), 10))
            ndcg_values.append(ndcg_at_k(ranked_ids, set(case.relevant_ids), 10))
            if case.version:
                version_values.append(
                    float(
                        bool(results)
                        and all(
                            version_is_compatible(case.version, result.document.version)
                            for result in results[:5]
                        )
                    )
                )
        ablations[mode] = {
            "evaluated_answerable_cases": len(answerable),
            "recall_at_5": _mean(recall_values),
            "mrr_at_10": _mean(reciprocal_values),
            "ndcg_at_10": _mean(ndcg_values),
            "version_accuracy": _mean(version_values),
        }

    true_positive = false_positive = false_negative = 0
    answered = 0
    valid_citations = 0
    answers_with_relevant_evidence = 0
    reason_matches = 0
    failures: list[dict[str, str]] = []
    validator = CitationValidator()
    for case in cases:
        response = service.diagnose(
            DiagnoseRequest(
                query=case.query,
                project=case.project,
                version=case.version,
                environment=EnvironmentInfo(),
            )
        )
        predicted_refusal = response.status == "refused"
        expected_refusal = not case.answerable
        true_positive += int(predicted_refusal and expected_refusal)
        false_positive += int(predicted_refusal and not expected_refusal)
        false_negative += int(not predicted_refusal and expected_refusal)
        answered += int(not predicted_refusal)
        if response.status == "answered":
            answers_with_relevant_evidence += int(
                bool({citation.document_id for citation in response.citations} & case.relevant_ids)
            )
            indexed_results = retriever.search(
                case.query,
                project=case.project,
                version=case.version,
                top_k=5,
                mode="hybrid_rerank",
            )
            valid_citations += int(validator.validate(response.citations, indexed_results))
        if case.expected_reason:
            reason_matches += int(response.reason_code == case.expected_reason)
        if predicted_refusal != expected_refusal:
            failures.append(
                {
                    "case_id": case.id,
                    "expected": "refused" if expected_refusal else "answered",
                    "actual": response.status,
                    "reason_code": response.reason_code,
                }
            )

    refusal = binary_scores(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    reason_case_count = sum(bool(case.expected_reason) for case in cases)
    agent_metrics = {
        "evaluated_cases": len(cases),
        "answered_cases": answered,
        "refused_cases": len(cases) - answered,
        "refusal_precision": refusal["precision"],
        "refusal_recall": refusal["recall"],
        "refusal_f1": refusal["f1"],
        "citation_validity": round(valid_citations / answered, 4) if answered else 0.0,
        "answer_evidence_recall": (
            round(answers_with_relevant_evidence / len(answerable), 4) if answerable else 0.0
        ),
        "expected_reason_accuracy": (
            round(reason_matches / reason_case_count, 4) if reason_case_count else 0.0
        ),
    }
    report: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": str(resolved.evaluation_path.relative_to(resolved.root)),
            "knowledge_base": str(resolved.data_path.relative_to(resolved.root)),
            "document_count": service.document_count,
            "case_count": len(cases),
            "reranker_weights": str(resolved.reranker_weights_path.relative_to(resolved.root)),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "scope": "curated offline smoke benchmark; not a production claim",
        },
        "retrieval_ablations": ablations,
        "agent": agent_metrics,
        "failures": failures,
    }
    report["thresholds"] = threshold_results(report)
    return report


def threshold_results(report: dict[str, Any]) -> dict[str, Any]:
    hybrid = report["retrieval_ablations"]["hybrid_rerank"]
    agent = report["agent"]
    checks = {
        "recall_at_5_gte_0_80": hybrid["recall_at_5"] >= 0.80,
        "mrr_at_10_gte_0_75": hybrid["mrr_at_10"] >= 0.75,
        "version_accuracy_eq_1": hybrid["version_accuracy"] == 1.0,
        "citation_validity_eq_1": agent["citation_validity"] == 1.0,
        "answer_evidence_recall_gte_0_90": agent["answer_evidence_recall"] >= 0.90,
        "refusal_f1_gte_0_80": agent["refusal_f1"] >= 0.80,
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
