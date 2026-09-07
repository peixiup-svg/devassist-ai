"""Citation construction and grounding validation."""

from __future__ import annotations

from devassist.domain import RetrievalResult
from devassist.models import Citation
from devassist.retrieval.text import compact_evidence


class CitationValidator:
    def build(self, results: list[RetrievalResult], limit: int = 3) -> list[Citation]:
        if limit <= 0:
            return []
        citations: list[Citation] = []
        seen: set[str] = set()
        for result in results:
            document = result.document
            if document.id in seen:
                continue
            evidence = compact_evidence(document.content)
            if not evidence or not document.url.startswith(("https://", "http://")):
                continue
            citations.append(
                Citation(
                    document_id=document.id,
                    title=document.title,
                    url=document.url,
                    version=document.version,
                    source_type=document.source_type,
                    evidence=evidence,
                )
            )
            seen.add(document.id)
            if len(citations) >= limit:
                break
        return citations

    @staticmethod
    def validate(citations: list[Citation], results: list[RetrievalResult]) -> bool:
        indexed = {result.document.id: result.document for result in results}
        if not citations:
            return False
        for citation in citations:
            document = indexed.get(citation.document_id)
            if document is None:
                return False
            if (
                citation.url != document.url
                or citation.title != document.title
                or citation.version != document.version
                or citation.source_type != document.source_type
            ):
                return False
            compact_original = " ".join(document.content.split())
            compact_evidence_text = " ".join(citation.evidence.rstrip("…").split())
            if compact_evidence_text not in compact_original:
                return False
        return True
