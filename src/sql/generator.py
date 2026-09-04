"""
SQL Pipeline: Natural Language to SQL Generator
Sends the user question + schema context to the LLM and returns a single
clean PostgreSQL/SQLite SELECT statement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import get_llm


_SYSTEM_PROMPT = """\
You are an expert SQL engineer for a Human Resources analytics system.

Your only job is to convert the user's natural-language question into a \
single, safe, read-only SELECT query using the schema provided.

Rules you MUST follow:
1. Output ONLY the raw SQL — no markdown fences, no explanation, no commentary.
2. Use only the tables and columns present in the schema below.
3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
4. Always add a LIMIT clause — default to 100 rows unless the question implies a scalar.
5. Use table aliases for clarity when joining multiple tables.
6. For date arithmetic in SQLite use: date('now', '-N months') or julianday().
7. Return NULL-safe comparisons; prefer IS NULL over = NULL.
8. For location/city queries (e.g. department locations like 'Austin, TX' or 'New York, NY'), use case-insensitive partial matching (e.g. d.location LIKE '%Austin%') rather than exact equality, since stored locations may include state codes or extra details.
9. If prior conversation context is provided, use it ONLY to resolve pronouns or omitted filters (e.g. department names or locations) for the current question.
10. If the question cannot be answered with the given schema, write exactly:
   -- CANNOT_GENERATE: <brief reason>
{history_context}
Schema:
{schema}
"""

_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class SQLGenerationResult:
    sql: str
    is_generatable: bool
    reason: str  # "ok" or the CANNOT_GENERATE message


class SQLGenerator:
    """Calls the LLM to produce a SQL SELECT from natural language."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._llm = get_llm(provider=provider, model_name=model, temperature=0.0)

    @staticmethod
    def _format_history_context(history: Optional[List[Any]]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-6:]:
            if isinstance(turn, dict):
                role = str(turn.get("role", "user")).capitalize()
                content = str(turn.get("content", "")).strip()
                if content and not any(p in content for p in ("SELECT ", "[Orchestrator]", "[Generated SQL]")):
                    lines.append(f"{role}: {content[:200]}")
            elif isinstance(turn, str) and turn.strip():
                lines.append(f"User: {turn.strip()[:200]}")
        if not lines:
            return ""
        return "\nRecent Conversation Context:\n" + "\n".join(lines) + "\n"

    @traceable(name="SQLGeneration", run_type="chain")
    def generate(
        self,
        question: str,
        schema_context: str,
        history: Optional[List[Any]] = None,
    ) -> SQLGenerationResult:
        """
        Generate SQL from a natural-language question and schema context.

        Returns a SQLGenerationResult with `is_generatable=False` if the LLM
        signals it cannot produce a valid query.
        """
        hist_ctx = self._format_history_context(history)
        system = SystemMessage(content=_SYSTEM_PROMPT.format(schema=schema_context, history_context=hist_ctx))
        human = HumanMessage(content=question)

        response = self._llm.invoke([system, human])
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_text = "".join(
                chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
                for chunk in raw_content
            )
        else:
            raw_text = str(raw_content)

        raw = raw_text.strip()


        sql = self._extract_sql(raw)

        if sql.lstrip().startswith("-- CANNOT_GENERATE"):
            prefix = "-- CANNOT_GENERATE:"
            if prefix in sql:
                reason = sql[sql.find(prefix) + len(prefix):].strip()
            else:
                reason = sql.replace("-- CANNOT_GENERATE", "").strip()
            return SQLGenerationResult(sql="", is_generatable=False, reason=reason)

        return SQLGenerationResult(sql=sql, is_generatable=True, reason="ok")


    @staticmethod
    def _extract_sql(raw: str) -> str:
        """Strip markdown code fences if the LLM wrapped its output."""
        match = _CODE_FENCE_RE.search(raw)
        if match:
            return match.group(1).strip()
        return raw
