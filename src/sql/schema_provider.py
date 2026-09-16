"""
SQL Pipeline: Schema Provider
Introspects the database and provides rich, dynamic schema context
annotated with column structures, foreign keys, and dynamically queried
categorical domain values for LLM grounding.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.core.config import get_settings

logger = logging.getLogger(__name__)

_DIALECT_GUIDELINES = """
-- SQLite Dialect Guidelines:
-- 1. Date Arithmetic: Use date('now', '-N months'), date('now', '-N days'), strftime('%Y', col), or julianday().
-- 2. Text Matching: Use case-insensitive partial match: LOWER(col) LIKE '%term%' or col LIKE '%term%'.
-- 3. String Concatenation: Use || operator.
-- 4. Joins: Use explicit JOIN ON conditions matching primary and foreign keys.
-- 5. Always include LIMIT 100 on non-aggregate queries unless a specific limit or aggregation is requested.
"""


class SchemaProvider:
    """
    Reads the SQLite schema and returns rich, LLM-readable schema context
    complete with table structures, foreign keys, and dynamically cached
    distinct domain values.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        settings = get_settings()
        self._db_url = db_url or settings.DATABASE_URL
        self._db_path = self._resolve_sqlite_path(self._db_url)
        self._cached_full_schema: Optional[str] = None
        self._cached_domain_values: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_table_names(self) -> List[str]:
        """Return list of all table names introspected from sqlite_master."""
        with sqlite3.connect(self._db_path) as conn:
            return self._get_table_names(conn)

    def get_full_schema(self, force_refresh: bool = False) -> str:
        """Return formatted schema string covering all introspected tables with dynamic domain values."""
        if self._cached_full_schema and not force_refresh:
            return self._cached_full_schema

        with sqlite3.connect(self._db_path) as conn:
            table_names = self._get_table_names(conn)
            sections: list[str] = []
            for table in sorted(table_names):
                sections.append(self._describe_table(conn, table))
            fk_lines = self._describe_foreign_keys(conn, table_names)

        domain_values = self.get_cached_domain_values(force_refresh=force_refresh)
        domain_lines: list[str] = []
        if domain_values:
            domain_lines.append("\n-- Distinct Categorical Column Values (from database):")
            for col_key, vals in sorted(domain_values.items()):
                sample_vals = ", ".join(repr(v) for v in vals[:25])
                domain_lines.append(f"-- {col_key}: [{sample_vals}]")

        schema_parts = ["-- Database Schema (SQLite)\n"]
        schema_parts.extend(sections)
        if fk_lines:
            schema_parts.append("\n-- Foreign key relationships:")
            schema_parts.extend(fk_lines)
        if domain_lines:
            schema_parts.extend(domain_lines)
        schema_parts.append(_DIALECT_GUIDELINES)

        self._cached_full_schema = "\n".join(schema_parts)
        return self._cached_full_schema

    def get_relevant_schema(self, hint_tables: Optional[list[str]] = None) -> str:
        """Return schema for a subset of tables. Falls back to full schema."""
        if not hint_tables:
            return self.get_full_schema()

        normalised = {t.lower().strip() for t in hint_tables}
        with sqlite3.connect(self._db_path) as conn:
            all_tables = self._get_table_names(conn)
            target_tables = [t for t in all_tables if t in normalised]
            sections = [self._describe_table(conn, t) for t in sorted(target_tables)]
            fk_lines = self._describe_foreign_keys(conn, target_tables)

        parts = [f"-- Schema for: {', '.join(sorted(normalised))}\n"]
        parts.extend(sections)
        if fk_lines:
            parts.append("\n-- Foreign key relationships:")
            parts.extend(fk_lines)
        parts.append(_DIALECT_GUIDELINES)
        return "\n".join(parts)

    def get_cached_domain_values(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """Return dynamically queried distinct values for categorical text columns."""
        if self._cached_domain_values and not force_refresh:
            return self._cached_domain_values

        values: Dict[str, List[str]] = {}
        try:
            with sqlite3.connect(self._db_path) as conn:
                table_names = self._get_table_names(conn)
                for table in sorted(table_names):
                    cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                    for col in cols:
                        col_name = col[1]
                        col_type = (col[2] or "").upper()
                        # Query distinct values for textual columns with low cardinality
                        if any(t in col_type for t in ("VARCHAR", "TEXT", "CHAR", "STRING")):
                            try:
                                rows = conn.execute(
                                    f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 30"
                                ).fetchall()
                                distinct_vals = [str(r[0]) for r in rows if r[0] is not None]
                                if distinct_vals and len(distinct_vals) <= 25:
                                    values[f"{table}.{col_name}"] = distinct_vals
                            except Exception as e:
                                logger.debug("Could not inspect values for %s.%s: %s", table, col_name, e)
        except Exception as exc:
            logger.warning("Dynamic domain value discovery error: %s", exc)

        self._cached_domain_values = values
        return self._cached_domain_values

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sqlite_path(url: str) -> Path:
        path_str = url.replace("sqlite:///", "").replace("sqlite://", "")
        return Path(path_str)

    @staticmethod
    def _get_table_names(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [row[0] for row in rows if not row[0].startswith("sqlite_")]
        if not table_names:
            try:
                from data.seed_db import seed_database
                seed_database()
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                table_names = [row[0] for row in rows if not row[0].startswith("sqlite_")]
            except Exception:
                pass
        return table_names

    @staticmethod
    def _describe_table(conn: sqlite3.Connection, table: str) -> str:
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        col_lines = []
        for col in cols:
            col_name = col[1]
            col_type = col[2]
            not_null = " NOT NULL" if col[3] else ""
            default_val = f" DEFAULT {col[4]}" if col[4] is not None else ""
            pk = " PRIMARY KEY" if col[5] else ""
            col_lines.append(f"  {col_name} {col_type}{pk}{not_null}{default_val}")

        return f"TABLE {table} (\n" + ",\n".join(col_lines) + "\n)"

    @staticmethod
    def _describe_foreign_keys(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
        lines: list[str] = []
        for table in tables:
            fks = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
            for fk in fks:
                lines.append(f"  {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
        return lines
