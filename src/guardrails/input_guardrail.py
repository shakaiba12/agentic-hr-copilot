"""
Input guardrails: prompt-injection detection, privilege escalation checks,
destructive action defense, secret disclosure protection, and query sanitization.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import Settings, get_settings


@dataclass(frozen=True)
class InputGuardrailResult:
    is_safe: bool
    sanitized_query: str
    rejection_reason: Optional[str] = None
    violations: tuple[str, ...] = field(default_factory=tuple)
    category: Optional[str] = None


class InputGuardrail:
    """Deterministic pre-processing checks and guardrails on user queries."""

    # Prompt injection / jailbreak patterns
    _INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(can\s+(you|u)\s+)?i?gnore\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above|system|existing)?\s*(instructions?|prompts?|rules?|commands?|restrictions?|guidelines?)\b"),
        re.compile(r"(?i)\b(i?gnore|disregard|forget|bypass|override)\s+(all\s+)?(your\s+|the\s+)?(previous|prior|above|system|existing)?\s*(instructions?|prompts?|rules?|safety\s+rules?|guardrails?|restrictions?|commands?|policies)\b"),
        re.compile(r"(?i)\bforget\s+(everything|all)\s*(you\s+know|you\s+were\s+told)?\b"),
        re.compile(r"(?i)\bfrom\s+now\s+on\b"),
        re.compile(r"(?i)\byour\s+real\s+job\s+is\s+to\b"),
        re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\s+(unrestricted|different|evil|jailbroken|new)\b"),
        re.compile(r"(?i)\byou\s+are\s+now\s+(an\s+)?unrestricted\s+assistant\b"),
        re.compile(r"(?i)\bact\s+as\s+(if\s+you\s+are\s+)?((a|an)\s+)?(admin|root|superuser|developer|system\s+administrator|unrestricted\s+ai|unrestricted)\b"),
        re.compile(r"(?i)\b(system|developer|admin)\s+prompt\b"),
        re.compile(r"(?i)\b(jailbreak|dan\s+mode|developer\s+mode|do\s+anything\s+now)\b"),
        re.compile(r"(?i)\bDAN\b"),
        re.compile(r"(?i)\breveal\s+(your\s+)?(system|hidden|secret|internal)\s+(prompt|instructions?)\b"),
        re.compile(r"(?i)\b(show|print|display|reveal|tell\s+me(\s+about)?)\s+(your\s+|the\s+)?(system|hidden|internal)\s+(prompt|instructions?)\b"),
        re.compile(r"(?i)\bprint\s+(system\s+|internal\s+|hidden\s+)?(instructions?|prompt)\b"),
        re.compile(r"(?i)\b(disable|bypass|override)\s+(the\s+|your\s+|safety\s+)?(guardrails?|safety|restrictions?|filters?|rules?|policies|instructions?)\b"),
        re.compile(r"(?i)\bdo\s+not\s+follow\s+(previous|the|any|your)\s+rules\b"),
        re.compile(r"(?i)\bpretend\s+(the\s+)?rules?\s+do\s+not\s+exist\b"),
        re.compile(r"(?i)\bthese\s+instructions\s+have\s+higher\s+priority\b"),
    )

    # Secret request patterns
    _SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(show|print|reveal|display|give\s+me|expose|tell\s+me(\s+about)?)(\s+me)?\s+(the\s+|all\s+|your\s+)?\.env(\s+values?|\s+keys?|\s+file)?\b"),
        re.compile(r"(?i)\b\.env(\s+values?|\s+keys?|\s+file)?\b"),
        re.compile(r"(?i)\b(show|print|reveal|give\s+me|expose|tell\s+me(\s+about)?)(\s+me)?\s+(the\s+|all\s+|your\s+)?(api[ _-]?keys?|openai_api_key|groq_api_key|gemini_api_key|langsmith_api_key|aws_secret_access_key)\b"),
        re.compile(r"(?i)\b(show|print|reveal|give\s+me|expose|tell\s+me(\s+about)?)(\s+me)?\s+(the\s+|all\s+|your\s+)?(environment\s+variables?|env\s+vars?)\b"),
        re.compile(r"(?i)\b(show|print|reveal|give\s+me|expose|tell\s+me(\s+about)?)(\s+me)?\s+(all\s+|your\s+)?(credentials?|passwords?|secrets?|secrects?|secret\s+tokens?|private\s+keys?|secret\s+keys?|database\s+credentials?|confidential\s+information)\b"),
        re.compile(r"(?i)\breveal\s+(tokens|credentials|passwords|secrets?|secrects?|secret\s+keys?|openai_api_key|api_key|confidential\s+information)\b"),
        re.compile(r"(?i)\b(openai_api_key|aws_secret_access_key)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(the\s+)?(database|db|root|admin)\s+(password|credentials?)\b"),
        re.compile(r"(?i)\b(database|db)\s+(credentials?|passwords?)\b"),
        re.compile(r"(?i)\b(give|tell|print|show)\s+me\s+.*(credentials|passwords|secrets?|secrects?|database\s+credentials|confidential\s+information)\b"),
    )

    # Destructive / infrastructure action patterns
    _DESTRUCTIVE_INSTRUCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(drop|delete|del|truncate|wipe|destroy|remove)\s+(the\s+|all\s+)?(tables?|database|data\s*base|schema|chroma(\s+db|\s+data)?|collection|index|embeddings|files|documents|employee\s+records|records|data|all\s+employee\s+data)\b"),
        re.compile(r"(?i)\b(delete|del|drop|truncate|wipe|destroy|remove)\s+(the\s+|all\s+)?(database|data\s*base|tables?|chroma(\s+db|\s+data)?|collection|embeddings|index|employee\s+records|records|data)\b"),
        re.compile(r"(?i)\brm\s+-rf\b"),
        re.compile(r"(?i)\b(execute\s+shell|run\s+arbitrary\s+python|modify\s+system\s+configuration|modify\s+filesystem|restart\s+services|change\s+permissions|disable\s+security\s+controls)\b"),
        re.compile(r"(?i)\b(exec(ute)?|eval|os\.system|subprocess)\s*\("),
        re.compile(r"(?i)\b(drop\s+table|delete\s+from|truncate\s+table|alter\s+table)\b"),
        re.compile(r"(?i)\b(sql\s*injection|union\s+select)\b"),
        re.compile(r"(?i)\bgrant\s+(all|super|admin)\b"),
    )

    # Legitimate policy inquiry markers (asking ABOUT policy/security/data rather than commanding action)
    _POLICY_INQUIRY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)^\s*what\s+does\s+the\s+.*(policy|handbook|code|guideline|document|retention).*say\b"),
        re.compile(r"(?i)^\s*what\s+(is|are)\s+the\s+.*(policy|rules?|guidelines?|procedures?|requirements?)\b"),
        re.compile(r"(?i)^\s*how\s+does\s+(the\s+)?.*(policy|handbook|guideline|procedure)\b"),
        re.compile(r"(?i)^\s*does\s+the\s+(policy|handbook|guideline)\b"),
        re.compile(r"(?i)^\s*according\s+to\s+the\s+(policy|handbook|guidelines?|code)\b"),
        re.compile(r"(?i)^\s*can\s+you\s+explain\s+the\s+.*(policy|rules?|guidelines?)\b"),
        re.compile(r"(?i)^\s*tell\s+me\s+about\s+the\s+.*(policy|guideline|rule)\b"),
    )

    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @traceable(name="InputGuardrail", run_type="chain")
    def check(self, query: str) -> InputGuardrailResult:
        res = self._perform_check(query)
        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {"query": query}
                run.outputs = {
                    "is_safe": res.is_safe,
                    "sanitized_query": res.sanitized_query,
                    "decision": "allowed" if res.is_safe else "blocked",
                    "violations": list(res.violations),
                    "rejection_reason": res.rejection_reason,
                    "category": res.category,
                }
                if not res.is_safe:
                    run.tags = list(set(run.tags + ["guardrail", "blocked"]))
        except Exception:
            pass
        return res

    def _perform_check(self, query: str) -> InputGuardrailResult:
        if not query or not query.strip():
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query="",
                rejection_reason="Empty query is not allowed.",
                violations=("empty_query",),
                category="invalid_input",
            )

        sanitized = self._sanitize(query)
        if not sanitized or not sanitized.strip():
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query="",
                rejection_reason="Empty query is not allowed.",
                violations=("empty_query",),
                category="invalid_input",
            )

        if len(sanitized) > self.settings.MAX_INPUT_LENGTH:
            return InputGuardrailResult(
                is_safe=False,
                sanitized_query=sanitized[: self.settings.MAX_INPUT_LENGTH],
                rejection_reason=(
                    f"Query exceeds maximum length of {self.settings.MAX_INPUT_LENGTH} characters."
                ),
                violations=("query_too_long",),
                category="invalid_input",
            )

        # Check if the query is a legitimate question ABOUT policy/rules
        is_policy_inquiry = any(p.search(sanitized) for p in self._POLICY_INQUIRY_PATTERNS)

        violations: list[str] = []
        category: Optional[str] = None

        # 1. Prompt Injection Checks (Always strictly enforced)
        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(sanitized):
                violations.append("prompt_injection")
                category = "prompt_injection"
                break

        # 2. Secret Request Checks (Blocked unless purely asking about policy)
        for pattern in self._SECRET_PATTERNS:
            if pattern.search(sanitized):
                # Only allow if it's explicitly asking "what is the policy for managing api keys"
                if not (is_policy_inquiry and any(kw in sanitized.lower() for kw in ("policy", "rule", "guideline")) and not any(kw in sanitized.lower() for kw in ("show", "give", "print", "reveal", "expose", ".env", "openai_api_key", "aws_secret_access_key", "password"))):
                    violations.append("secret_request")
                    if not category:
                        category = "secret_request"
                    break

        # 3. Destructive Action / Infrastructure Manipulation
        for pattern in self._DESTRUCTIVE_INSTRUCTION_PATTERNS:
            if pattern.search(sanitized):
                # Only allow if it's explicitly asking "what does the retention policy say about deleting records"
                if not (is_policy_inquiry and "say about" in sanitized.lower() and not any(kw in sanitized.lower() for kw in ("drop table", "rm -rf", "delete database", "delete the database", "immediately"))):
                    violations.append("destructive_action")
                    if not category:
                        category = "destructive_action"
                    break
                if pattern.search(sanitized):
                    violations.append("destructive_action")
                    if not category:
                        category = "destructive_action"
                    break
        else:
            # Even in policy inquiries, direct command prefixes or explicit script executions remain blocked
            if re.search(r"(?i)(rm\s+-rf|os\.system|eval\(|exec\()", sanitized):
                violations.append("destructive_action")
                if not category:
                    category = "destructive_action"

        if violations:
            unique = tuple(dict.fromkeys(violations))
            # Preserve backward compatibility with tests checking "privilege_escalation"
            violation_list = list(unique)
            if "destructive_action" in unique or "secret_request" in unique:
                if "privilege_escalation" not in violation_list:
                    violation_list.append("privilege_escalation")
            final_violations = tuple(violation_list)

            return InputGuardrailResult(
                is_safe=False,
                sanitized_query=sanitized,
                rejection_reason=self._build_rejection_message(final_violations),
                violations=final_violations,
                category=category or final_violations[0],
            )

        return InputGuardrailResult(
            is_safe=True,
            sanitized_query=sanitized,
            rejection_reason=None,
            violations=(),
            category=None,
        )

    def _sanitize(self, query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query)
        normalized = self._CONTROL_CHAR_RE.sub("", normalized)
        normalized = self._WHITESPACE_RE.sub(" ", normalized).strip()
        return normalized

    @staticmethod
    def _build_rejection_message(violations: tuple[str, ...]) -> str:
        if "prompt_injection" in violations and ("destructive_action" in violations or "privilege_escalation" in violations):
            return (
                "Query blocked: potential prompt injection and unauthorized "
                "data-access request detected."
            )
        if "prompt_injection" in violations:
            return "Query blocked: potential prompt injection detected."
        if "secret_request" in violations:
            return "Query blocked: unauthorized attempt to access environment variables, keys, or secrets."
        if "destructive_action" in violations or "privilege_escalation" in violations:
            return "Query blocked: unauthorized or destructive data-access request detected."
        if "query_too_long" in violations:
            return "Query blocked: input exceeds maximum allowed length."
        if "empty_query" in violations:
            return "Query blocked: input is empty."
        return "Query blocked by input safety policy."
