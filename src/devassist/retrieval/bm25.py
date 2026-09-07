"""Dependency-free BM25 implementation tuned for errors and code symbols."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from devassist.retrieval.text import tokenise


class BM25Index:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1 = k1
        self.b = b
        self._tokens = [tokenise(document) for document in documents]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self._tokens)
        self._idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def score(self, query: str, index: int) -> float:
        if index < 0 or index >= len(self._tokens):
            raise IndexError(index)
        query_tokens = tokenise(query)
        if not query_tokens or not self._tokens[index]:
            return 0.0
        frequencies = self._frequencies[index]
        length = self._lengths[index]
        normaliser = self.k1 * (1.0 - self.b + self.b * length / max(self._average_length, 1.0))
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency:
                score += self._idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + normaliser)
                )
        return score

    def search(
        self,
        query: str,
        *,
        candidate_indices: Sequence[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        indices = candidate_indices if candidate_indices is not None else range(len(self._tokens))
        scored = [(index, self.score(query, index)) for index in indices]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
