"""PII redaction middleware -- detects and masks personal data in agent outputs.

Applies regex-based detection for common PII patterns (SSN, email,
phone numbers, credit card numbers) and replaces them with masked
placeholders.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

_MASKS: dict[str, str] = {
    "ssn": "***-**-****",
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "credit_card": "****-****-****-****",
}


class PIIRedactionMiddleware:
    """Detects and redacts PII from text and structured data.

    Usage::

        redactor = PIIRedactionMiddleware()
        clean_text = redactor.redact_text("Call 555-123-4567")
        # => "Call [REDACTED_PHONE]"

        clean_dict = redactor.redact_dict({"email": "user@example.com"})
        # => {"email": "[REDACTED_EMAIL]"}
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._redaction_count = 0

    def redact(self, text: str) -> str:
        """Alias for :meth:`redact_text` for convenience."""
        return self.redact_text(text)

    def redact_text(self, text: str) -> str:
        """Redact all detected PII patterns from *text*.

        Returns the redacted text with PII replaced by masked
        placeholders.
        """
        if not self.enabled or not text:
            return text

        result = text
        for pii_type, pattern in _PII_PATTERNS.items():
            mask = _MASKS[pii_type]
            matches = pattern.findall(result)
            if matches:
                self._redaction_count += len(matches)
                logger.debug(
                    "pii_detected",
                    pii_type=pii_type,
                    count=len(matches),
                )
                result = pattern.sub(mask, result)

        return result

    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively redact PII from a dictionary.

        Walks all string values and nested dicts/lists, redacting
        detected PII in place.
        """
        if not self.enabled:
            return data
        return self._walk(data)  # type: ignore[return-value]

    def detect_pii(self, text: str) -> list[dict[str, Any]]:
        """Detect PII patterns in *text* without redacting.

        Returns a list of dicts with ``type``, ``match``, and ``start``/
        ``end`` positions.
        """
        detections: list[dict[str, Any]] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            for match in pattern.finditer(text):
                detections.append(
                    {
                        "type": pii_type,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return detections

    @property
    def redaction_count(self) -> int:
        """Total number of PII items redacted so far."""
        return self._redaction_count

    def reset_count(self) -> None:
        """Reset the redaction counter."""
        self._redaction_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk(self, obj: Any) -> Any:
        """Recursively walk a data structure and redact strings."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {k: self._walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk(item) for item in obj]
        return obj
