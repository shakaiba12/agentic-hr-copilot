"""
SQL guardrails: SELECT-only enforcement, table allowlists, and injection blocking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import sqlparse
from sqlparse.sql import Identifier

from src.core.config import Settings, get_settings


@dataclass(frozen=True)
class SQLGuardrailResult:
    is_valid: bool
    normalized_sql: str
    notes: str
    referenced_tables: tuple[str, ...] = field(default_factory=tuple)
    violations: tuple[str, ...] = field(default_factory=tuple)


class SQLGuardrail:
    """Deterministic SQL safety validator before execution."""

    _MULTI_STATEMENT_RE = re.compile(r";\s*\S")
    _COMMENT_RE = re.compile(r"(?i)(/\*.*?\*/|--[^\n]*)")
    _STRING_LITERAL_RE = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")
    _FROM_JOIN_TABLE_RE = re.compile(
        r"(?i)\b(?:FROM|JOIN)\s+([`\"[]?[a-zA-Z_][\w$]*[`\"]]?)"
        r"(?:\.([`\"[]?[a-zA-Z_][\w$]*[`\"]]?))?"
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._allowed_tables = {t.lower() for t in self.settings.ALLOWED_SQL_TABLES}
        self._blocked_keywords = {k.upper() for k in self.settings.BLOCKED_SQL_KEYWORDS}

    def validate(self, sql: str) -> SQLGuardrailResult:
        if not sql or not sql.strip():
            return SQLGuardrailResult(
                is_valid=False,
                normalized_sql="",
                notes="SQL query is empty.",
                violations=("empty_sql",),
            )

        normalized = self._normalize(sql)
        violations: list[str] = []

        if self._MULTI_STATEMENT_RE.search(normalized):
            violations.append("multiple_statements")

        stripped_for_scan = self._strip_literals_and_comments(normalized)
        violations.extend(self._scan_blocked_keywords(stripped_for_scan))

        statements = sqlparse.parse(normalized)
        if not statements:
            violations.append("unparseable_sql")
            return self._fail(normalized, violations)

        if len(statements) > 1:
            violations.append("multiple_statements")

        statement = statements[0]
        stmt_type = statement.get_type()
        if stmt_type != "SELECT":
            violations.append("not_select_only")

        referenced_tables = self._extract_tables(statement)
        disallowed = sorted(t for t in referenced_tables if t not in self._allowed_tables)
        if disallowed:
            violations.append("disallowed_table")

        if violations:
            return self._fail(normalized, violations, referenced_tables)

        return SQLGuardrailResult(
            is_valid=True,
            normalized_sql=normalized,
            notes="SQL passed safety validation.",
            referenced_tables=referenced_tables,
            violations=(),
        )

    def _normalize(self, sql: str) -> str:
        normalized = sql.strip().rstrip(";")
        return sqlparse.format(
            normalized,
            strip_comments=True,
            reindent=False,
            keyword_case="upper",
        )

    def _strip_literals_and_comments(self, sql: str) -> str:
        without_comments = self._COMMENT_RE.sub(" ", sql)
        return self._STRING_LITERAL_RE.sub(" ", without_comments)

    def _scan_blocked_keywords(self, sql: str) -> list[str]:
        violations: list[str] = []
        tokens = re.findall(r"\b[A-Z_]+\b", sql.upper())
        for token in tokens:
            if token in self._blocked_keywords:
                violations.append(f"blocked_keyword:{token.lower()}")
        return violations

    def _extract_tables(self, statement) -> tuple[str, ...]:
        sql_without_literals = self._strip_literals_and_comments(str(statement))
        tables: set[str] = set()

        for match in self._FROM_JOIN_TABLE_RE.finditer(sql_without_literals):
            schema_or_table = match.group(1)
            table_only = match.group(2)
            raw_name = table_only or schema_or_table
            name = self._clean_identifier(raw_name)
            if name:
                tables.add(name.lower())

        if tables:
            return tuple(sorted(tables))

        return self._extract_tables_from_tokens(statement)

    @staticmethod
    def _clean_identifier(identifier: str) -> str:
        cleaned = identifier.strip().strip("[]`\"")
        return cleaned.split(".")[-1]

    def _extract_tables_from_tokens(self, statement) -> tuple[str, ...]:
        """Fallback token walk limited to FROM/JOIN table positions."""
        tables: set[str] = set()
        from sqlparse.tokens import Keyword

        tokens = list(statement.flatten())
        capture_next_identifier = False

        for token in tokens:
            if token.ttype is Keyword and token.normalized in {"FROM", "JOIN"}:
                capture_next_identifier = True
                continue

            if capture_next_identifier:
                if token.ttype is Keyword:
                    capture_next_identifier = False
                    continue
                if isinstance(token.parent, Identifier):
                    name = self._identifier_name(token.parent)
                    if name:
                        tables.add(name.lower())
                    capture_next_identifier = False

        return tuple(sorted(tables))

    @staticmethod
    def _identifier_name(identifier: Identifier) -> Optional[str]:
        real_name = identifier.get_real_name()
        if real_name:
            return real_name.split(".")[-1]

        name = identifier.get_name()
        if name:
            return name.split(".")[-1]

        return None

    def _fail(
        self,
        normalized_sql: str,
        violations: list[str],
        referenced_tables: tuple[str, ...] = (),
    ) -> SQLGuardrailResult:
        unique = tuple(dict.fromkeys(violations))
        return SQLGuardrailResult(
            is_valid=False,
            normalized_sql=normalized_sql,
            notes=self._build_notes(unique, referenced_tables),
            referenced_tables=referenced_tables,
            violations=unique,
        )

    def _build_notes(
        self,
        violations: tuple[str, ...],
        referenced_tables: tuple[str, ...],
    ) -> str:
        messages: list[str] = []

        if "empty_sql" in violations:
            messages.append("SQL query is empty.")
        if "multiple_statements" in violations:
            messages.append("Only a single SQL statement is permitted.")
        if "not_select_only" in violations:
            messages.append("Only SELECT queries are permitted.")
        if "unparseable_sql" in violations:
            messages.append("SQL could not be parsed safely.")
        if "disallowed_table" in violations:
            disallowed = sorted(t for t in referenced_tables if t not in self._allowed_tables)
            messages.append(
                f"Query references disallowed tables: {', '.join(disallowed)}."
            )
        blocked = [v.split(":", 1)[1] for v in violations if v.startswith("blocked_keyword:")]
        if blocked:
            messages.append(
                f"Query contains blocked keywords: {', '.join(sorted(set(blocked)))}."
            )

        return " ".join(messages) if messages else "SQL failed safety validation."
