"""Thread-safe JSONL knowledge store with atomic reload semantics."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from devassist.domain import KnowledgeDocument


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._documents: tuple[KnowledgeDocument, ...] = ()
        self._by_id: dict[str, KnowledgeDocument] = {}
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def documents(self) -> tuple[KnowledgeDocument, ...]:
        with self._lock:
            return self._documents

    def get(self, document_id: str) -> KnowledgeDocument | None:
        with self._lock:
            return self._by_id.get(document_id)

    def read(self) -> tuple[KnowledgeDocument, ...]:
        """Parse and validate the configured file without mutating live state."""

        if not self.path.exists():
            raise FileNotFoundError(f"knowledge base not found: {self.path}")

        loaded: list[KnowledgeDocument] = []
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {self.path}:{line_number}: {exc.msg}"
                    ) from exc
                document = KnowledgeDocument.from_mapping(payload)
                if not document.id:
                    raise ValueError(f"empty document id at {self.path}:{line_number}")
                if document.id in seen:
                    raise ValueError(f"duplicate document id: {document.id}")
                if not document.url.startswith(("https://", "http://")):
                    raise ValueError(f"document URL must be http(s): {document.id}")
                seen.add(document.id)
                loaded.append(document)

        if not loaded:
            raise ValueError(f"knowledge base is empty: {self.path}")

        return tuple(loaded)

    def replace(self, documents: tuple[KnowledgeDocument, ...]) -> int:
        """Atomically expose an already validated document snapshot."""

        if not documents:
            raise ValueError("cannot replace the store with an empty snapshot")
        by_id = {document.id: document for document in documents}
        if len(by_id) != len(documents):
            raise ValueError("cannot replace the store with duplicate document ids")
        with self._lock:
            self._documents = documents
            self._by_id = by_id
            self._generation += 1
            return self._generation

    def load(self) -> int:
        return self.replace(self.read())
