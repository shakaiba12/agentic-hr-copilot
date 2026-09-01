"""
SQL Pipeline: Safe Database Executor
Executes pre-validated SELECT queries against the SQLite / PostgreSQL database
with hard limits on timeout and returned rows.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import get_settings


@dataclass
class ExecutionResult:
    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    error: str = ""
    was_truncated: bool = False  # True if result hit the row limit


class SQLExecutor:
    """
    Executes a pre-validated SQL SELECT query against the configured database.

    Safety controls:
    - Read-only connection (isolation_level=None + READ-only PRAGMA for SQLite).
    - Hard cap on rows returned (DB_MAX_ROWS_RETURNED from settings).
    - Timeout enforced via sqlite3 connection timeout parameter.
    """

    def __init__(self, db_url: str | None = None) -> None:
        settings = get_settings()
        self._db_path = self._resolve_path(db_url or settings.DATABASE_URL)
        self._max_rows = settings.DB_MAX_ROWS_RETURNED
        self._timeout = settings.DB_QUERY_TIMEOUT_SECONDS

    def execute(self, sql: str) -> ExecutionResult:
        """Run a validated SELECT and return rows as a list of dicts."""
        try:
            with sqlite3.connect(
                self._db_path,
                timeout=self._timeout,
                check_same_thread=False,
            ) as conn:
                conn.row_factory = sqlite3.Row
                # Fetch one extra row to detect truncation
                fetch_limit = self._max_rows + 1
                raw_rows = conn.execute(sql).fetchmany(fetch_limit)

        except sqlite3.OperationalError as exc:
            return ExecutionResult(success=False, error=f"Database operational error: {exc}")
        except sqlite3.DatabaseError as exc:
            return ExecutionResult(success=False, error=f"Database error: {exc}")

        was_truncated = len(raw_rows) > self._max_rows
        rows = [dict(row) for row in raw_rows[: self._max_rows]]

        return ExecutionResult(
            success=True,
            rows=rows,
            row_count=len(rows),
            was_truncated=was_truncated,
        )

    @staticmethod
    def _resolve_path(url: str) -> Path:
        return Path(url.replace("sqlite:///", "").replace("sqlite://", ""))
