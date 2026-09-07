from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from devassist.config import Settings
from devassist.evaluation.metrics import (
    binary_scores,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from devassist.evaluation.runner import (
    evaluate,
    load_cases,
    threshold_results,
    write_report,
)


def test_retrieval_metrics_match_hand_calculated_values() -> None:
    ranked = ["irrelevant", "relevant-b", "relevant-a"]
    relevant = {"relevant-a", "relevant-b"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant, 10) == 0.5
    expected_dcg = 1 / 1.584962500721156 + 1 / 2
    expected_ideal = 1 + 1 / 1.584962500721156
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(expected_dcg / expected_ideal)


def test_retrieval_metrics_handle_empty_relevance_and_cutoff() -> None:
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert reciprocal_rank(["a"], {"a"}, 0) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], {"a"}, 0) == 0.0


def test_ndcg_does_not_reward_duplicate_document_ids() -> None:
    score = ndcg_at_k(["relevant", "relevant"], {"relevant"}, 2)

    assert score == 1.0
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (
            {"true_positive": 8, "false_positive": 2, "false_negative": 2},
            {"precision": 0.8, "recall": 0.8, "f1": 0.8},
        ),
        (
            {"true_positive": 0, "false_positive": 0, "false_negative": 0},
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        ),
        (
            {"true_positive": 1, "false_positive": 2, "false_negative": 0},
            {"precision": 0.3333, "recall": 1.0, "f1": 0.5},
        ),
    ],
)
def test_binary_scores(counts: dict[str, int], expected: dict[str, float]) -> None:
    assert binary_scores(**counts) == expected


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def valid_case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "case-1",
        "query": "PyTorch torch.load error",
        "project": "PyTorch",
        "version": "V2.6",
        "answerable": True,
        "relevant_ids": ["doc-1"],
        "tags": ["smoke"],
    }
    case.update(overrides)
    return case


def test_load_cases_normalises_and_builds_immutable_values(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "cases.jsonl", [valid_case()])

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].project == "pytorch"
    assert cases[0].version == "v2.6"
    assert cases[0].relevant_ids == frozenset({"doc-1"})
    assert cases[0].tags == ("smoke",)


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "cases.jsonl", [valid_case()])
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert len(load_cases(path)) == 1


def test_load_cases_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_cases(path)


def test_load_cases_rejects_duplicate_and_inconsistent_cases(tmp_path: Path) -> None:
    duplicate_path = write_jsonl(
        tmp_path / "duplicate.jsonl",
        [valid_case(), valid_case(query="different query")],
    )
    with pytest.raises(ValueError, match="duplicate evaluation case id"):
        load_cases(duplicate_path)

    answerable_without_relevance = write_jsonl(
        tmp_path / "inconsistent-answerable.jsonl",
        [valid_case(answerable=True, relevant_ids=[])],
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_cases(answerable_without_relevance)

    refusal_with_relevance = write_jsonl(
        tmp_path / "inconsistent-refusal.jsonl",
        [valid_case(answerable=False, relevant_ids=["doc-1"])],
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_cases(refusal_with_relevance)


def test_load_cases_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation set is empty"):
        load_cases(path)


def passing_report() -> dict[str, object]:
    return {
        "retrieval_ablations": {
            "hybrid_rerank": {
                "recall_at_5": 0.8,
                "mrr_at_10": 0.75,
                "version_accuracy": 1.0,
            }
        },
        "agent": {
            "citation_validity": 1.0,
            "answer_evidence_recall": 0.9,
            "refusal_f1": 0.8,
        },
    }


def test_threshold_results_report_each_gate() -> None:
    passed = threshold_results(passing_report())

    assert passed["passed"] is True
    assert all(passed["checks"].values())

    failed_report = passing_report()
    failed_report["agent"]["refusal_f1"] = 0.79  # type: ignore[index]
    failed = threshold_results(failed_report)
    assert failed["passed"] is False
    assert failed["checks"]["refusal_f1_gte_0_80"] is False


def test_write_report_is_atomic_utf8_and_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    report = {"status": "通过", "thresholds": {"passed": True}}

    write_report(report, target)

    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert not target.with_suffix(".json.tmp").exists()


def test_evaluate_runs_reproducibly_against_tmp_data(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory()

    first = evaluate(settings)
    second = evaluate(settings)

    assert first["metadata"]["case_count"] == 2
    assert first["metadata"]["document_count"] == 4
    assert set(first["retrieval_ablations"]) == {
        "bm25",
        "dense",
        "hybrid",
        "hybrid_rerank",
    }
    assert first["retrieval_ablations"] == second["retrieval_ablations"]
    assert first["agent"] == second["agent"]
    assert first["failures"] == second["failures"]
    assert isinstance(first["thresholds"]["passed"], bool)


def test_evaluate_does_not_count_empty_results_as_version_correct(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory()
    settings.evaluation_path.write_text(
        json.dumps(
            {
                "id": "empty-retrieval",
                "query": "☃☃☃",
                "project": "pytorch",
                "version": "2.6",
                "answerable": True,
                "relevant_ids": ["pytorch-load-26"],
                "tags": ["regression"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(settings)

    assert all(
        metrics["version_accuracy"] == 0.0 for metrics in report["retrieval_ablations"].values()
    )
