"""Tokenisation, version matching, and safe evidence formatting."""

from __future__ import annotations

import re
from collections.abc import Iterable

_PART_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:+/-]*|\d+(?:\.\d+){0,3}|[\u3400-\u9fff]+")
_VERSION_RE = re.compile(r"(?<!\d)v?(\d+(?:\.\d+){1,3}(?:rc\d+)?(?:\+[\w.]+)?)", re.I)
_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "i",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
    "为什么",
    "怎么",
    "如何",
    "一个",
    "这个",
    "可以",
    "需要",
    "多少",
    "什么",
    "时候",
}


def tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    for part in _PART_RE.findall(text):
        lowered = part.lower()
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            if len(part) > 1:
                tokens.append(part)
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
            else:
                tokens.append(part)
        elif lowered not in _STOPWORDS:
            tokens.append(lowered)
            if "." in lowered:
                tokens.extend(piece for piece in lowered.split(".") if piece)
    return [token for token in tokens if token and token not in _STOPWORDS]


def unique_tokens(text: str) -> set[str]:
    return set(tokenise(text))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = unique_tokens(left)
    right_tokens = unique_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def extract_exception_names(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_EXCEPTION_RE.findall(text)))


def extract_version(text: str) -> str | None:
    match = _VERSION_RE.search(text)
    return match.group(1).lower() if match else None


def version_parts(version: str | None) -> tuple[int, ...] | None:
    if not version or version.lower() in {"all", "latest", "any"}:
        return None
    match = re.match(r"^v?(\d+(?:\.\d+){0,3})", version.strip(), re.I)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def version_compatibility(requested: str | None, document_version: str) -> float:
    if not requested:
        return 1.0
    if document_version.lower() in {"all", "any", "latest"}:
        return 0.96
    wanted = version_parts(requested)
    actual = version_parts(document_version)
    if wanted is None or actual is None:
        return 0.75
    common = min(len(wanted), len(actual))
    if wanted[:common] == actual[:common]:
        if len(wanted) >= 2 and len(actual) >= 2 and wanted[:2] == actual[:2]:
            return 1.0
        return 0.92
    if wanted[0] == actual[0]:
        return 0.68
    return 0.35


def version_is_compatible(requested: str | None, document_version: str) -> bool:
    return version_compatibility(requested, document_version) >= 0.9


def compact_evidence(text: str, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def stable_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
