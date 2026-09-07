"""Train safe blended weights from labelled positive and hard-negative pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from devassist.retrieval.hybrid import HybridRetriever
from devassist.retrieval.reranker import DEFAULT_WEIGHTS, FeatureReranker
from devassist.store import KnowledgeStore

FEATURE_NAMES = tuple(DEFAULT_WEIGHTS)


@dataclass(frozen=True, slots=True)
class TrainingPair:
    query: str
    document_id: str
    label: int
    requested_version: str | None = None


def load_pairs(path: Path) -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            label = int(payload["label"])
            if label not in {0, 1}:
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            pairs.append(
                TrainingPair(
                    query=str(payload["query"]),
                    document_id=str(payload["document_id"]),
                    label=label,
                    requested_version=(
                        str(payload["requested_version"])
                        if payload.get("requested_version")
                        else None
                    ),
                )
            )
    if {pair.label for pair in pairs} != {0, 1}:
        raise ValueError("training pairs must contain both positive and negative labels")
    return pairs


def train_reranker(
    *,
    knowledge_path: Path,
    pairs_path: Path,
    output_path: Path,
    blend: float = 0.30,
) -> dict[str, Any]:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between 0 and 1")
    store = KnowledgeStore(knowledge_path)
    store.load()
    retriever = HybridRetriever(store.documents)
    feature_builder = FeatureReranker()
    pairs = load_pairs(pairs_path)
    by_id = {document.id: document for document in store.documents}

    rows: list[list[float]] = []
    labels: list[int] = []
    for pair in pairs:
        document = by_id.get(pair.document_id)
        if document is None:
            raise ValueError(f"unknown training document: {pair.document_id}")
        candidates = retriever.search(
            pair.query,
            project=document.project,
            version=pair.requested_version,
            top_k=len(store.documents),
            mode="hybrid",
        )
        fused_by_id = {result.document.id: result.score for result in candidates}
        features = feature_builder.features(
            pair.query,
            document,
            fused_by_id.get(document.id, 0.0),
            pair.requested_version,
        )
        rows.append([features[name] for name in FEATURE_NAMES])
        labels.append(pair.label)

    matrix: NDArray[np.float64] = np.asarray(rows, dtype=np.float64)
    targets: NDArray[np.int_] = np.asarray(labels, dtype=np.int_)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=42,
        solver="liblinear",
    ).fit(matrix, targets)
    positive: NDArray[np.float64] = np.asarray(
        np.clip(classifier.coef_[0], 0.0, None), dtype=np.float64
    )
    if float(positive.sum()) == 0.0:
        learned: NDArray[np.float64] = np.asarray(
            [DEFAULT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64
        )
    else:
        learned = positive / positive.sum()
    defaults: NDArray[np.float64] = np.asarray(
        [DEFAULT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    blended = (1.0 - blend) * defaults + blend * learned
    blended = blended / blended.sum()
    weights = {
        name: round(float(value), 8) for name, value in zip(FEATURE_NAMES, blended, strict=True)
    }
    accuracy = float(classifier.score(matrix, targets))
    payload: dict[str, Any] = {
        **weights,
        "_metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "training_pairs": len(pairs),
            "positive_pairs": int(targets.sum()),
            "hard_negative_pairs": int(len(targets) - targets.sum()),
            "training_accuracy": round(accuracy, 4),
            "blend_with_safe_defaults": blend,
            "warning": "Training accuracy is a pipeline check, not a held-out quality claim.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload
