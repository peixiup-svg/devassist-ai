"""Deterministic input analysis and bounded tool routing."""

from __future__ import annotations

import re

from devassist.domain import QueryAnalysis
from devassist.models import EnvironmentInfo
from devassist.projects import canonical_project
from devassist.retrieval.text import extract_exception_names, extract_version
from devassist.security import inspect_query

KNOWN_PROJECTS: dict[str, tuple[str, ...]] = {
    "pytorch": ("pytorch", "torch", "cuda", "dataloader", "state_dict"),
    "transformers": ("transformers", "huggingface", "tokenizer", "trainer"),
    "fastapi": ("fastapi", "uvicorn", "pydantic", "422"),
}
_CANONICAL_PROJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "pytorch": ("pytorch",),
    "transformers": ("transformers", "huggingface"),
    "fastapi": ("fastapi",),
}
_PROJECT_HINT_WEIGHTS: dict[str, dict[str, int]] = {
    "pytorch": {"torch": 2, "cuda": 1, "dataloader": 2, "state_dict": 2},
    "transformers": {"tokenizer": 2, "trainer": 1},
    "fastapi": {"uvicorn": 2, "pydantic": 1, "422": 1},
}

_VERSION_RE = re.compile(r"(?<!\d)v?(\d+(?:\.\d+){1,3}(?:rc\d+)?(?:\+[\w.]+)?)", re.I)
_PROJECT_VERSION_ALIASES: dict[str, tuple[str, ...]] = {
    "pytorch": ("pytorch", "torch"),
    "transformers": ("transformers", "huggingface"),
    "fastapi": ("fastapi",),
}
_VERSION_LABELS: dict[str, tuple[str, ...]] = {
    **_PROJECT_VERSION_ALIASES,
    "python": ("python",),
    "cuda": ("cuda",),
    "pydantic": ("pydantic",),
    "uvicorn": ("uvicorn",),
}


class QueryAnalyzer:
    def analyse(
        self,
        query: str,
        *,
        project: str | None,
        version: str | None,
        environment: EnvironmentInfo,
    ) -> QueryAnalysis:
        clean_query, safety_warnings = inspect_query(query)
        lowered = clean_query.lower()
        detected_project = canonical_project(project) if project else self._detect_project(lowered)
        detected_version = version or self._version_from_environment(detected_project, environment)
        detected_version = detected_version or self._version_from_query(
            detected_project, clean_query
        )
        exceptions = extract_exception_names(clean_query)

        source_types: tuple[str, ...]
        if exceptions or any(term in lowered for term in ("报错", "error", "traceback", "失败")):
            intent = "diagnosis"
            # A runtime error can be introduced by a release change, so diagnosis
            # must not discard changelog/migration evidence before retrieval.
            source_types = ("documentation", "issue", "changelog", "migration")
        elif any(
            term in lowered for term in ("升级", "迁移", "deprecated", "migration", "changelog")
        ):
            intent = "migration"
            source_types = ("migration", "changelog", "documentation")
        elif any(term in lowered for term in ("兼容", "版本", "compatible", "version")):
            intent = "compatibility"
            source_types = ("documentation", "changelog", "migration")
        else:
            intent = "documentation"
            source_types = ("documentation", "issue", "changelog", "migration")

        missing: list[str] = []
        if detected_project and not detected_version:
            missing.append(f"{detected_project} 版本")
        if exceptions and not environment.python_version:
            missing.append("Python 版本")
        if "cuda" in lowered and not environment.cuda_version:
            missing.append("CUDA 版本")
        if exceptions and not environment.operating_system:
            missing.append("操作系统")

        return QueryAnalysis(
            clean_query=clean_query,
            project=detected_project,
            version=detected_version,
            intent=intent,
            exception_names=exceptions,
            source_types=source_types,
            missing_information=tuple(missing),
            safety_warnings=safety_warnings,
        )

    @staticmethod
    def _detect_project(lowered_query: str) -> str | None:
        canonical_matches: list[tuple[int, str]] = []
        for project, aliases in _CANONICAL_PROJECT_ALIASES.items():
            for alias in aliases:
                match = re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered_query)
                if match:
                    canonical_matches.append((match.start(), project))
        if canonical_matches:
            return min(canonical_matches, key=lambda item: item[0])[1]

        scores = {
            project: sum(
                weight
                for alias, weight in aliases.items()
                if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered_query)
            )
            for project, aliases in _PROJECT_HINT_WEIGHTS.items()
        }
        project, score = max(scores.items(), key=lambda item: item[1])
        return project if score else None

    @staticmethod
    def _version_from_environment(project: str | None, environment: EnvironmentInfo) -> str | None:
        if not project:
            return None
        package_aliases = {
            "pytorch": _PROJECT_VERSION_ALIASES["pytorch"],
            "transformers": ("transformers",),
            # Dependency versions are not framework versions. In particular,
            # Pydantic 2.x must never be interpreted as FastAPI 2.x.
            "fastapi": _PROJECT_VERSION_ALIASES["fastapi"],
        }
        for package in package_aliases.get(project, (project,)):
            if package in environment.packages:
                return re.sub(r"^[=~^<>!\s]+", "", environment.packages[package])
        return None

    @staticmethod
    def _version_from_query(project: str | None, query: str) -> str | None:
        """Return a version tied to the detected project, not the first number.

        Error reports commonly contain Python, CUDA and framework versions in
        the same sentence. A nearest-label match prevents Python 3.12 or CUDA
        12.1 from being used as the requested PyTorch version. If a lone version
        has no nearby competing label, it remains a useful fallback.
        """

        if not project:
            return extract_version(query)

        lowered = query.lower()
        labels: list[tuple[int, int, str]] = []
        for owner, aliases in _VERSION_LABELS.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", re.I)
                labels.extend(
                    (match.start(), match.end(), owner) for match in pattern.finditer(lowered)
                )

        unlabelled: list[str] = []
        project_matches: list[tuple[int, int, str]] = []
        for order, match in enumerate(_VERSION_RE.finditer(query)):
            nearby: list[tuple[int, str]] = []
            for start, end, owner in labels:
                if end <= match.start():
                    gap = match.start() - end
                elif match.end() <= start:
                    gap = start - match.end()
                else:
                    gap = 0
                if gap <= 24:
                    nearby.append((gap, owner))
            if nearby:
                distance, owner = min(nearby, key=lambda item: item[0])
                if owner == project:
                    project_matches.append((distance, order, match.group(1).lower()))
            else:
                unlabelled.append(match.group(1).lower())

        if project_matches:
            return min(project_matches, key=lambda item: (item[0], item[1]))[2]
        return unlabelled[0] if len(unlabelled) == 1 else None
