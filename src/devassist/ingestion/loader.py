"""UTF-8 JSON Lines helpers for the offline ingestion pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class JsonlError(ValueError):
    """Raised when a JSONL input cannot be decoded or validated."""


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from *path* with useful line-numbered errors.

    Empty lines are intentionally ignored. UTF-8 with an optional BOM is accepted,
    which makes files exported by common Windows editors work without conversion.
    """

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    message = f"invalid JSON in {source} at line {line_number}: {exc.msg}"
                    raise JsonlError(message) from exc
                if not isinstance(value, dict):
                    message = (
                        f"expected a JSON object in {source} at line {line_number}, "
                        f"got {type(value).__name__}"
                    )
                    raise JsonlError(message)
                yield value
    except UnicodeDecodeError as exc:
        raise JsonlError(f"{source} is not valid UTF-8: {exc}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all JSON objects from a UTF-8 JSONL file."""

    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write mappings as deterministic UTF-8 JSON Lines.

    The temporary file is created beside the destination so ``Path.replace`` stays
    atomic on Windows and does not cross filesystem boundaries.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                serialized = json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write(serialized)
                handle.write("\n")
            handle.flush()
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
