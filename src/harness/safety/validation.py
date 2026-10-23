"""Validation of user input at the trust boundary."""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters except tab, newline, carriage return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Unicode "tag" characters (U+E0000-U+E007F) can smuggle invisible instructions.
_TAGS = re.compile("[\U000e0000-\U000e007f]")


class InputValidationError(ValueError):
    """User input failed validation. The message is safe to return to the client."""


def validate_user_input(text: str, *, max_chars: int = 20_000) -> str:
    """Normalize and bound user input; raises `InputValidationError` when invalid."""
    if not isinstance(text, str):
        raise InputValidationError("input must be a string")
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = _TAGS.sub("", _CONTROL.sub("", cleaned)).strip()
    if not cleaned:
        raise InputValidationError("input is empty")
    if len(cleaned) > max_chars:
        raise InputValidationError(f"input exceeds {max_chars} characters")
    return cleaned
