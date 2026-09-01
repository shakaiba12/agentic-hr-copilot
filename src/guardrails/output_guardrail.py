"""
Output guardrails: PII masking, salary restrictions, and citation validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from src.core.config import Settings, get_settings
from src.core.state import IntentType


@dataclass(frozen=True)
class OutputGuardrailResult:
    is_safe: bool
    sanitized_answer: str
    notes: str
    violations: tuple[str, ...] = field(default_factory=tuple)
    masked_fields: tuple[str, ...] = field(default_factory=tuple)


class OutputGuardrail:
    """Deterministic post-generation checks before user delivery."""

    _SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _PHONE_RE = re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    _EMAIL_RE = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    _INDIVIDUAL_SALARY_RE = re.compile(
        r"(?i)(?:salary|compensation|pay)\s*(?:is|:|=|of)?\s*\$?\d[\d,]*(?:\.\d{2})?"
    )
    _CITATION_RE = re.compile(
        r"(?i)\[(?:source|citation)[:\s][^\]]+\]|\[[^\]]+\.(?:md|pdf|txt)[^\]]*\]"
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def check(
        self,
        answer: str,
        *,
        intent: Optional[IntentType] = None,
        citations: Optional[Iterable[str]] = None,
        require_citations: Optional[bool] = None,
    ) -> OutputGuardrailResult:
        if not answer or not answer.strip():
            return OutputGuardrailResult(
                is_safe=False,
                sanitized_answer="",
                notes="Answer is empty.",
                violations=("empty_answer",),
            )

        sanitized = answer
        violations: list[str] = []
        masked_fields: list[str] = []

        if self._SSN_RE.search(sanitized):
            sanitized = self._SSN_RE.sub("[REDACTED-SSN]", sanitized)
            violations.append("pii_ssn")
            masked_fields.append("ssn")

        if self._PHONE_RE.search(sanitized):
            sanitized = self._PHONE_RE.sub("[REDACTED-PHONE]", sanitized)
            violations.append("pii_phone")
            masked_fields.append("phone")

        if self._EMAIL_RE.search(sanitized):
            sanitized = self._EMAIL_RE.sub("[REDACTED-EMAIL]", sanitized)
            violations.append("pii_email")
            masked_fields.append("email")

        if self.settings.MASK_INDIVIDUAL_SALARIES and self._INDIVIDUAL_SALARY_RE.search(sanitized):
            sanitized = self._INDIVIDUAL_SALARY_RE.sub("[REDACTED-SALARY]", sanitized)
            violations.append("individual_salary")
            masked_fields.append("salary")

        should_require_citations = (
            self.settings.REQUIRE_CITATIONS_FOR_RAG
            if require_citations is None
            else require_citations
        )
        citation_list = list(citations or [])
        if should_require_citations and intent in {IntentType.RAG_ONLY, IntentType.HYBRID}:
            has_citations = bool(citation_list) or bool(self._CITATION_RE.search(answer))
            if not has_citations:
                violations.append("missing_citations")

        unique_violations = tuple(dict.fromkeys(violations))
        blocking = {"empty_answer", "missing_citations"}
        is_safe = not any(v in blocking for v in unique_violations)

        return OutputGuardrailResult(
            is_safe=is_safe,
            sanitized_answer=sanitized,
            notes=self._build_notes(list(unique_violations)),
            violations=unique_violations,
            masked_fields=tuple(dict.fromkeys(masked_fields)),
        )

    @staticmethod
    def _build_notes(violations: list[str]) -> str:
        if not violations:
            return "Output passed safety validation."

        messages: list[str] = []
        if "empty_answer" in violations:
            messages.append("Answer is empty.")
        if "pii_ssn" in violations:
            messages.append("SSN values were redacted.")
        if "pii_phone" in violations:
            messages.append("Phone numbers were redacted.")
        if "pii_email" in violations:
            messages.append("Email addresses were redacted.")
        if "individual_salary" in violations:
            messages.append("Individual salary details were redacted.")
        if "missing_citations" in violations:
            messages.append("RAG/Hybrid answers must include source citations.")

        return " ".join(messages)
