"""
SQL Pipeline: Schema Provider
Introspects the live database and returns a rich, query-relevant schema context
string that is fed to the SQL generator as grounding context.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from src.core.config import get_settings


class SchemaProvider:
    """
    Reads the SQLite / PostgreSQL schema and returns a concise, LLM-readable
    representation of tables, columns, types, and foreign keys.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        settings = get_settings()
        self._db_url = db_url or settings.DATABASE_URL
        self._db_path = self._resolve_sqlite_path(self._db_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_full_schema(self) -> str:
        """Return a formatted schema string covering all allowed tables."""
        with sqlite3.connect(self._db_path) as conn:
            table_names = self._get_table_names(conn)
            sections: list[str] = []
            for table in sorted(table_names):
                sections.append(self._describe_table(conn, table))
            fk_lines = self._describe_foreign_keys(conn, table_names)

        schema_parts = ["-- HR Database Schema\n"]
        schema_parts.extend(sections)
        if fk_lines:
            schema_parts.append("\n-- Foreign key relationships")
            schema_parts.extend(fk_lines)
        return "\n".join(schema_parts)

    def get_relevant_schema(self, hint_tables: Optional[list[str]] = None) -> str:
        """
        Return schema for a subset of tables.
        Falls back to the full schema when no hints are provided.
        """
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
            parts.append("\n-- Foreign key relationships")
            parts.extend(fk_lines)
        return "\n".join(parts)

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
        table_names = [row[0] for row in rows]
        if not table_names:
            try:
                from data.seed_db import seed_database
                seed_database()
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                table_names = [row[0] for row in rows]
            except Exception:
                pass
        return table_names


    @staticmethod
    def _describe_table(conn: sqlite3.Connection, table: str) -> str:
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        col_lines = [
            f"  {col[1]} {col[2]}{'  NOT NULL' if col[3] else ''}{'  DEFAULT ' + str(col[4]) if col[4] is not None else ''}"
            for col in cols
        ]
        return f"TABLE {table} (\n" + ",\n".join(col_lines) + "\n)"

    @staticmethod
    def _describe_foreign_keys(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
        lines: list[str] = []
        for table in tables:
            fks = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
            for fk in fks:
                lines.append(
                    f"  {table}.{fk[3]} -> {fk[2]}.{fk[4]}"
                )
        return lines
