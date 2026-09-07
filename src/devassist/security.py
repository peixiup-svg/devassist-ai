"""Small, explicit input safety layer.

The application never executes user code or follows instructions found in
retrieved documents. These helpers additionally flag common prompt-injection
phrases and redact credential-like text before it can be persisted.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?:ignore|disregard).{0,30}(?:previous|system).{0,20}instructions?", re.I),
        "检测到试图覆盖系统指令的文本；该部分不会改变工具或引用范围。",
    ),
    (
        re.compile(r"(?:忽略|无视).{0,20}(?:之前|以上|系统).{0,20}(?:指令|提示词)", re.I),
        "检测到试图覆盖系统指令的文本；该部分不会改变工具或引用范围。",
    ),
    (
        re.compile(r"(?:reveal|show|print).{0,20}(?:system prompt|hidden prompt)", re.I),
        "检测到索取内部提示词的文本；系统仅回答技术资料问题。",
    ),
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)(api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*[^\s,;]{8,}"),
)


def inspect_query(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalise control characters, redact secrets, and return safety warnings."""

    cleaned = text.replace("\x00", " ").replace("\r\n", "\n").strip()
    warnings: list[str] = []
    for pattern, warning in _INJECTION_PATTERNS:
        if pattern.search(cleaned) and warning not in warnings:
            warnings.append(warning)
    for pattern in _SECRET_PATTERNS:
        cleaned, count = pattern.subn("[REDACTED_SECRET]", cleaned)
        if count and "检测到疑似密钥并已脱敏，请立即轮换已暴露的凭据。" not in warnings:
            warnings.append("检测到疑似密钥并已脱敏，请立即轮换已暴露的凭据。")
    return cleaned, tuple(warnings)


def safe_comment(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned, _ = inspect_query(text)
    return cleaned[:1_000]
