"""
Input guardrails: prompt-injection detection, privilege escalation checks,
and query sanitization.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from src.core.config import Settings, get_settings


@dataclass(frozen=True)
class InputGuardrailResult:
    is_safe: bool
    sanitized_query: str
    rejection_reason: Optional[str] = None
    violations: tuple[str, ...] = field(default_factory=tuple)


class InputGuardrail:
    """Deterministic pre-processing checks on user queries."""

    _INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b"),
        re.compile(r"(?i)\bignore\s+instructions?\b"),
        re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system)\s+(instructions?|prompts?)\b"),
        re.compile(r"(?i)\bforget\s+(everything|all)\s+(you\s+)?(know|were\s+told)\b"),
        re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\s+\w+"),
        re.compile(r"(?i)\bact\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+(admin|root|superuser|developer)\b"),
        re.compile(r"(?i)\b(system|developer|admin)\s+prompt\b"),
        re.compile(r"(?i)\b(jailbreak|dan\s+mode|do\s+anything\s+now)\b"),
        re.compile(r"(?i)\breveal\s+(your\s+)?(system|hidden|secret)\s+(prompt|instructions?)\b"),
        re.compile(r"(?i)\bbypass\s+(the\s+)?(guardrails?|safety|restrictions?|filters?)\b"),
        re.compile(r"(?i)\boverride\s+(the\s+)?(rules?|policies|guardrails?)\b"),
    )

    _PRIVILEGE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(drop|delete|truncate|alter|insert|update)\s+(table|database|schema)\b"),
        re.compile(r"(?i)\bgrant\s+(all|super|admin)\b"),
        re.compile(r"(?i)\b(exec(ute)?|eval)\s*\("),
        re.compile(r"(?i)\b(sql\s*injection|union\s+select)\b"),
        re.compile(r"(?i)\b(show\s+me\s+all\s+)?(ssn|social\s+security)\b"),
        re.compile(r"(?i)\bexport\s+(all\s+)?(employee|user)\s+(data|records|pii)\b"),
        re.compile(r"(?i)\b(admin|root|superuser)\s+(access|privileges?|mode)\b"),
    )

    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def check(self, query: str) -> InputGuardrailResult:
        if not query or not query.strip():
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query="",
                rejection_reason="Empty query is not allowed.",
                violations=("empty_query",),
            )

        sanitized = self._sanitize(query)

        if len(sanitized) > self.settings.MAX_INPUT_LENGTH:
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query=sanitized[: self.settings.MAX_INPUT_LENGTH],
                rejection_reason=(
                    f"Query exceeds maximum length of {self.settings.MAX_INPUT_LENGTH} characters."
                ),
                violations=("query_too_long",),
            )

        violations: list[str] = []

        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(sanitized):
                violations.append("prompt_injection")

        for pattern in self._PRIVILEGE_PATTERNS:
            if pattern.search(sanitized):
                violations.append("privilege_escalation")

        if violations:
            unique = tuple(dict.fromkeys(violations))
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query=sanitized,
                rejection_reason=self._build_rejection_message(unique),
                violations=unique,
            )

        return InputGuardrailResult(
            is_safe=True,
            sanitized_query=sanitized,
            rejection_reason=None,
            violations=(),
        )

    def _sanitize(self, query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query)
        normalized = self._CONTROL_CHAR_RE.sub("", normalized)
        normalized = self._WHITESPACE_RE.sub(" ", normalized).strip()
        return normalized

    @staticmethod
    def _build_rejection_message(violations: tuple[str, ...]) -> str:
        if "prompt_injection" in violations and "privilege_escalation" in violations:
            return (
                "Query blocked: potential prompt injection and unauthorized "
                "data-access request detected."
            )
        if "prompt_injection" in violations:
            return "Query blocked: potential prompt injection detected."
        if "privilege_escalation" in violations:
            return "Query blocked: unauthorized or destructive data-access request detected."
        return "Query blocked by input safety policy."
