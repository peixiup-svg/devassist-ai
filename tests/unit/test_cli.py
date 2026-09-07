from __future__ import annotations

import json

from devassist.cli import build_parser, serialise_output
from devassist.models import DiagnoseResponse, SearchHit


def test_cli_parser_accepts_bounded_top_k() -> None:
    args = build_parser().parse_args(
        ["search", "torch.load weights_only", "--project", "torch", "--top-k", "1"]
    )

    assert args.command == "search"
    assert args.top_k == 1


def test_cli_serialises_search_result_lists_as_json() -> None:
    hit = SearchHit(
        document_id="doc-1",
        project="pytorch",
        version="2.6",
        source_type="documentation",
        title="torch.load",
        url="https://example.test/doc",
        excerpt="evidence",
        score=0.9,
        component_scores={"bm25_raw": 1.0},
        reasons=["bm25-rank-1"],
    )

    payload = json.loads(serialise_output([hit]))

    assert payload[0]["document_id"] == "doc-1"


def test_cli_serialises_diagnosis_models_as_json() -> None:
    response = DiagnoseResponse(
        request_id="request-fixed",
        status="refused",
        reason_code="NO_EVIDENCE",
        diagnosis="not enough evidence",
        steps=[],
        citations=[],
        confidence=0.0,
        missing_information=[],
        tools_used=[],
        safety_warnings=[],
        latency_ms=1.0,
    )

    assert json.loads(serialise_output(response))["reason_code"] == "NO_EVIDENCE"
