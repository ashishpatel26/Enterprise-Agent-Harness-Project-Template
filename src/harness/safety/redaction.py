"""Secret redaction for logs, audit records, events and tool outputs."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any, ClassVar

REDACTED = "[REDACTED]"

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),  # Anthropic
    re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),  # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),  # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),  # Google API key
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{16,}=*"),
    re.compile(r"(?i)(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),  # credentials in URLs
    re.compile(
        r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
        r"(\s*[:=]\s*)(['\"]?)[^\s'\",;]{4,}\3"
    ),
]
_SENSITIVE_KEYS = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key)"
)


class Redactor:
    def __init__(self, known_secrets: Iterable[str] = ()) -> None:
        # Exact values (configured keys) are redacted even if no pattern matches them.
        self._known = sorted({s for s in known_secrets if s and len(s) >= 6}, key=len, reverse=True)

    def add_secret(self, value: str) -> None:
        if value and len(value) >= 6 and value not in self._known:
            self._known.append(value)
            self._known.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._known:
            text = text.replace(secret, REDACTED)
        for pattern in _PATTERNS:
            if pattern.groups >= 3:
                text = pattern.sub(
                    lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}", text
                )
            else:
                text = pattern.sub(REDACTED, text)
        return text

    def redact_obj(self, obj: Any) -> Any:
        """Recursively redact strings; values under sensitive keys are fully replaced."""
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {
                k: (
                    REDACTED
                    if isinstance(k, str) and _SENSITIVE_KEYS.search(k) and v
                    else self.redact_obj(v)
                )
                for k, v in obj.items()
            }
        if isinstance(obj, list | tuple):
            return type(obj)(self.redact_obj(v) for v in obj)
        return obj


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the message, its args and structured extras."""

    _STD: ClassVar[set[str]] = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)))

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redactor.redact(record.msg)
        if record.args:
            record.args = tuple(
                self.redactor.redact_obj(a)
                for a in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        for key, value in list(vars(record).items()):
            if key not in self._STD:
                setattr(record, key, self.redactor.redact_obj(value))
        return True
