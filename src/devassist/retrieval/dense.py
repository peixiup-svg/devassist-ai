"""Offline semantic-ish dense retrieval using TF-IDF followed by LSA.

This backend is deterministic and model-download free, making it ideal for CI
and interviews. A sentence-transformer adapter can replace it without changing
the HybridRetriever contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LSADenseIndex:
    def __init__(self, documents: Sequence[str], dimensions: int = 128) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            norm="l2",
        )
        self._svd: TruncatedSVD | None = None
        self._matrix: NDArray[np.float64]
        if not documents:
            self._matrix = np.empty((0, 0), dtype=np.float64)
            return
        sparse = self._vectorizer.fit_transform(documents)
        component_count = min(dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
        if component_count >= 2:
            self._svd = TruncatedSVD(n_components=component_count, random_state=42)
            transformed = self._svd.fit_transform(sparse)
            self._matrix = np.asarray(normalize(transformed), dtype=np.float64)
        else:
            self._matrix = np.asarray(sparse.toarray(), dtype=np.float64)

    def scores(self, query: str) -> NDArray[np.float64]:
        if self._matrix.shape[0] == 0 or not query.strip():
            return np.zeros(self._matrix.shape[0], dtype=np.float64)
        query_vector = self._vectorizer.transform([query])
        if self._svd is not None:
            transformed = np.asarray(normalize(self._svd.transform(query_vector)), dtype=np.float64)
            scores = np.asarray(self._matrix @ transformed.T).reshape(-1)
        else:
            dense_query = np.asarray(query_vector.toarray(), dtype=np.float64)
            scores = np.asarray(self._matrix @ dense_query.T, dtype=np.float64).reshape(-1)
        return np.asarray(np.clip(scores, 0.0, 1.0), dtype=np.float64)

    def search(
        self,
        query: str,
        *,
        candidate_indices: Sequence[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        all_scores = self.scores(query)
        indices = candidate_indices if candidate_indices is not None else range(len(all_scores))
        ranked = [(index, float(all_scores[index])) for index in indices]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]
