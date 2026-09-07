"""Canonical project identifiers shared by analysis and retrieval."""

from __future__ import annotations

PROJECT_ALIASES: dict[str, str] = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "transformers": "transformers",
    "huggingface": "transformers",
    "hugging-face": "transformers",
    "fastapi": "fastapi",
}


def canonical_project(value: str | None) -> str | None:
    """Normalise supported aliases while preserving unknown filters."""

    if value is None:
        return None
    normalised = value.strip().lower()
    if not normalised:
        return None
    return PROJECT_ALIASES.get(normalised, normalised)
