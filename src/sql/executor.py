"""
SQL Pipeline: Safe Database Executor
Executes pre-validated SELECT queries against SQLite with:
- Dry-run compilation verification via EXPLAIN QUERY PLAN
- Read-only enforcement via PRAGMA query_only = ON
- Query execution timeout and row limits
- Detailed latency telemetry and column metadata
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import get_settings


@dataclass
class ExecutionResult:
    success: bool
    rows: List[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    column_names: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""
    was_truncated: bool = False  # True if result hit the row limit


class SQLExecutor:
    """
    Executes validated SQL SELECT queries against SQLite.

    Safety controls & Execution parameters:
    - Pre-execution dry-run via EXPLAIN QUERY PLAN to catch syntax and column errors.
    - Read-only connection enforcement via PRAGMA query_only = ON;
    - Row limit truncation: DB_MAX_ROWS_RETURNED.
    - SQLite Connection Busy Timeout: DB_QUERY_TIMEOUT_SECONDS (controls busy-handler wait on locks).
    - Latency telemetry and column metadata collection.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        settings = get_settings()
        self._db_path = self._resolve_path(db_url or settings.DATABASE_URL)
        self._max_rows = settings.DB_MAX_ROWS_RETURNED
        self._timeout = settings.DB_QUERY_TIMEOUT_SECONDS


    def dry_run(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL syntax, column names, aliases, and functions using SQLite EXPLAIN QUERY PLAN
        without actually reading data rows.

        Returns (is_valid, error_message).
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty."

        clean_sql = sql.strip().rstrip(";")
        try:
            with sqlite3.connect(
                self._db_path,
                timeout=self._timeout,
                check_same_thread=False,
            ) as conn:
                conn.execute("PRAGMA query_only = ON;")
                conn.execute(f"EXPLAIN QUERY PLAN {clean_sql}")
            return True, "ok"
        except sqlite3.OperationalError as exc:
            return False, f"SQLite syntax or schema error: {exc}"
        except sqlite3.DatabaseError as exc:
            return False, f"SQLite database error: {exc}"
        except Exception as exc:
            return False, f"Unexpected dry-run error: {exc}"

    @traceable(name="SQLExecution", run_type="chain")
    def execute(self, sql: str) -> ExecutionResult:
        """Run a validated SELECT and return rows as a list of dicts with timing."""
        res = self._perform_execute(sql)
        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {"sql": sql}
                run.outputs = {
                    "success": res.success,
                    "row_count": res.row_count,
                    "latency_ms": res.latency_ms,
                    "was_truncated": res.was_truncated,
                    "error": res.error,
                    "rows_sample": res.rows[:3] if res.rows else [],
                }
        except Exception:
            pass
        return res

    def _perform_execute(self, sql: str) -> ExecutionResult:
        clean_sql = sql.strip().rstrip(";")
        start_time = time.perf_counter()

        try:
            with sqlite3.connect(
                self._db_path,
                timeout=self._timeout,
                check_same_thread=False,
            ) as conn:
                conn.execute("PRAGMA query_only = ON;")
                conn.row_factory = sqlite3.Row

                # Fetch one extra row to detect truncation
                fetch_limit = self._max_rows + 1
                cursor = conn.execute(clean_sql)
                raw_rows = cursor.fetchmany(fetch_limit)
                col_names = [d[0] for d in cursor.description] if cursor.description else []

        except sqlite3.OperationalError as exc:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return ExecutionResult(
                success=False,
                error=f"Database operational error: {exc}",
                latency_ms=elapsed,
            )
        except sqlite3.DatabaseError as exc:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return ExecutionResult(
                success=False,
                error=f"Database error: {exc}",
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return ExecutionResult(
                success=False,
                error=f"Execution error: {exc}",
                latency_ms=elapsed,
            )

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        was_truncated = len(raw_rows) > self._max_rows
        rows = [dict(row) for row in raw_rows[: self._max_rows]]

        return ExecutionResult(
            success=True,
            rows=rows,
            row_count=len(rows),
            column_names=col_names,
            latency_ms=elapsed,
            was_truncated=was_truncated,
        )

    @staticmethod
    def _resolve_path(url: str) -> Path:
        return Path(url.replace("sqlite:///", "").replace("sqlite://", ""))
