from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from devassist.domain import KnowledgeDocument, RetrievalResult
from devassist.retrieval.bm25 import BM25Index
from devassist.retrieval.dense import LSADenseIndex
from devassist.retrieval.hybrid import HybridRetriever
from devassist.retrieval.reranker import DEFAULT_WEIGHTS, FeatureReranker


@pytest.mark.parametrize(
    ("k1", "b"),
    [(0.0, 0.75), (-1.0, 0.75), (1.5, -0.01), (1.5, 1.01)],
)
def test_bm25_rejects_invalid_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError, match="invalid BM25"):
        BM25Index(["document"], k1=k1, b=b)


def test_bm25_ranks_exact_error_and_code_symbol_first() -> None:
    index = BM25Index(
        [
            "RuntimeError Expected all tensors same device torch.nn.Module cuda",
            "General CUDA installation and driver guide",
            "Tokenizer padding documentation",
        ]
    )

    results = index.search("RuntimeError torch.nn.Module same device", top_k=2)

    assert results[0][0] == 0
    assert results[0][1] > results[1][1]


def test_bm25_handles_empty_queries_candidates_and_bounds() -> None:
    index = BM25Index(["alpha beta", "beta gamma"])

    assert index.search("", top_k=2) == [(0, 0.0), (1, 0.0)]
    assert index.search("alpha", top_k=0) == []
    assert [item[0] for item in index.search("beta", candidate_indices=[1], top_k=5)] == [1]
    with pytest.raises(IndexError):
        index.score("alpha", -1)
    with pytest.raises(IndexError):
        index.score("alpha", 2)


def test_bm25_empty_corpus_is_searchable() -> None:
    assert BM25Index([]).search("anything") == []


def test_dense_empty_index_and_blank_query_return_zero_scores() -> None:
    empty = LSADenseIndex([])
    populated = LSADenseIndex(["torch cuda", "tokenizer padding"])

    assert empty.search("torch") == []
    assert empty.scores("torch").shape == (0,)
    assert np.array_equal(populated.scores("   "), np.zeros(2))


def test_dense_ranks_semantically_similar_text_and_respects_candidates() -> None:
    index = LSADenseIndex(
        [
            "torch cuda gpu memory allocation error",
            "tokenizer padding uses eos token",
            "fastapi validates a pydantic request body",
        ],
        dimensions=8,
    )

    scores = index.scores("cuda memory")
    results = index.search("cuda memory", top_k=2)

    assert scores.shape == (3,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert results[0][0] == 0
    assert index.search("padding", candidate_indices=[1], top_k=10)[0][0] == 1
    assert index.search("padding", top_k=0) == []


def test_dense_single_document_uses_sparse_fallback() -> None:
    index = LSADenseIndex(["torch weights_only checkpoint"])

    assert index.search("weights_only", top_k=1)[0][1] > 0


def test_hybrid_validates_configuration_and_mode(
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        HybridRetriever(sample_documents, candidate_k=0)
    with pytest.raises(ValueError, match="positive"):
        HybridRetriever(sample_documents, rrf_constant=0)

    retriever = HybridRetriever(sample_documents)
    with pytest.raises(ValueError, match="unsupported retrieval mode"):
        retriever.search("torch load", mode="unknown")


def test_rrf_fusion_matches_hand_calculated_ranking(
    monkeypatch: pytest.MonkeyPatch,
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    documents = sample_documents[:3]
    retriever = HybridRetriever(documents, candidate_k=3, rrf_constant=60)
    monkeypatch.setattr(
        retriever.bm25,
        "search",
        lambda *_args, **_kwargs: [(0, 10.0), (1, 5.0)],
    )
    monkeypatch.setattr(
        retriever.dense,
        "search",
        lambda *_args, **_kwargs: [(1, 1.0), (2, 0.5)],
    )

    results = retriever.search("query", top_k=3, mode="hybrid")

    assert [result.document.id for result in results] == [
        "pytorch-load-25",
        "pytorch-load-26",
        "transformers-padding",
    ]
    assert results[0].score == pytest.approx(
        0.55 * ((1 / 62 + 1 / 61) / (2 / 61)) + 0.25 * 0.5 + 0.20
    )
    assert results[1].score == pytest.approx(0.55 * 0.5 + 0.25)
    assert results[2].score == pytest.approx(0.55 * ((1 / 62) / (2 / 61)) + 0.20 * 0.5)


def test_hybrid_filters_project_source_type_and_exact_version(
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents, candidate_k=10)

    results = retriever.search(
        "torch.load weights_only",
        project=" PYTORCH ",
        version="2.6",
        source_types=["changelog"],
        top_k=10,
        mode="hybrid",
    )

    assert [result.document.id for result in results] == ["pytorch-load-26"]
    assert all(result.component_scores["version_compatibility"] == 1.0 for result in results)


def test_hybrid_returns_explainable_fallback_only_when_no_version_is_compatible(
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents[:2], candidate_k=10)

    results = retriever.search(
        "torch.load weights_only", project="pytorch", version="2.7", mode="hybrid"
    )

    assert results
    assert all("version-fallback" in result.reasons for result in results)
    assert all(result.component_scores["version_compatibility"] < 0.9 for result in results)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_hybrid_modes_are_bounded_deterministic_and_deduplicated(
    mode: str,
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents, candidate_k=10)

    first = retriever.search("torch.load weights_only", top_k=3, mode=mode)
    second = retriever.search("torch.load weights_only", top_k=3, mode=mode)

    assert first == second
    assert len(first) <= 3
    assert len({result.document.id for result in first}) == len(first)
    assert all(0.0 <= result.score <= 1.0 for result in first)
    assert retriever.search(" ", mode=mode) == []
    assert retriever.search("valid", top_k=0, mode=mode) == []


def test_hybrid_returns_empty_for_unknown_filters(
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents)

    assert retriever.search("torch", project="unknown") == []
    assert retriever.search("torch", source_types=["unknown"]) == []


def test_hybrid_canonicalises_supported_project_aliases(
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents)

    results = retriever.search("torch.load weights_only", project=" Torch ")

    assert results
    assert all(result.document.project == "pytorch" for result in results)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_hybrid_does_not_rank_zero_score_candidates(
    mode: str,
    sample_documents: tuple[KnowledgeDocument, ...],
) -> None:
    retriever = HybridRetriever(sample_documents)

    assert retriever.search("☃☃☃", mode=mode) == []


def test_feature_reranker_loads_known_weights_and_ignores_unknowns(
    tmp_path: Path,
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"fused": 1.0, "unknown": 100.0}), encoding="utf-8")
    reranker = FeatureReranker(path)
    document = document_factory()
    result = RetrievalResult(document=document, score=0.5, reasons=("seed",))

    reranked = reranker.rerank(
        "PyTorch 2.6 torch.load UnpicklingError",
        [result],
        requested_version="2.6",
    )

    assert reranker.weights["fused"] == 1.0
    assert set(reranker.weights) == set(DEFAULT_WEIGHTS)
    assert reranked[0].component_scores["feature_version"] == 1.0
    assert "version-exact" in reranked[0].reasons
    assert "exact-code-or-error-token" in reranked[0].reasons
    assert 0.0 <= reranked[0].score <= 1.0


def test_reranker_ties_are_ordered_by_document_id(
    document_factory: Callable[..., KnowledgeDocument],
) -> None:
    first = RetrievalResult(document=document_factory(id="z"), score=0.2)
    second = RetrievalResult(document=document_factory(id="a"), score=0.2)

    results = FeatureReranker().rerank("unrelated", [first, second], requested_version=None)

    assert [result.document.id for result in results] == ["a", "z"]
