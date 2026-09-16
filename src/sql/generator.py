"""
SQL Pipeline: Natural Language to SQL Generator
Converts user natural-language questions + schema context into a single,
safe, read-only SQLite SELECT query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import get_llm
from src.sql.exemplars import format_exemplars_for_prompt, get_relevant_exemplars


_SYSTEM_PROMPT = """\
You are an expert SQL engineer for PeopleQuery AI.

Your job is to convert the user's natural-language question into a single, safe, read-only SQLite SELECT query strictly using the introspected database schema and domain values provided below.

Strict Rules:
1. Output ONLY the raw SQL query — no markdown fences, no conversational filler, no commentary.
2. Use only the tables, columns, and foreign keys explicitly defined in the provided schema.
3. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, ATTACH, DETACH, or PRAGMA.
4. Default to LIMIT 100 on non-aggregate queries unless the user asks for a specific count or scalar aggregation.
5. Use clear table aliases when joining tables based on foreign key relationships defined in the schema.
6. For date arithmetic in SQLite use: date('now', '-N months'), date('now', '-N days'), strftime('%Y', date_col), or julianday().
7. For text matching and filtering, use case-insensitive matching: LOWER(column) LIKE '%term%' or column LIKE '%term%'.
8. For categorical filters, match against the distinct column values provided in the schema context.
9. If prior conversation history is provided, use it ONLY to resolve pronouns or omitted contextual filters from previous turns.
10. If the question asks for data not present in the database schema, output exactly:
    -- CANNOT_GENERATE: <brief explanation of missing schema tables or columns>
{history_context}
{exemplars_context}
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
    """Calls the LLM to produce a SQLite SELECT query from natural language."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._llm = get_llm(provider=provider, model_name=model, temperature=0.0)

    @staticmethod
    def _format_history_context(history: Optional[List[Any]]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-4:]:
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

    def _build_system_prompt(
        self,
        schema_context: str,
        history: Optional[List[Any]],
        use_exemplars: bool,
        question: str,
        error_feedback: Optional[str],
    ) -> SystemMessage:
        hist_ctx = self._format_history_context(history)
        exemplars_ctx = ""
        if use_exemplars and not error_feedback:
            matched_exemplars = get_relevant_exemplars(question, max_exemplars=2)
            exemplars_ctx = format_exemplars_for_prompt(matched_exemplars)

        prompt_str = _SYSTEM_PROMPT.format(
            schema=schema_context,
            history_context=hist_ctx,
            exemplars_context=exemplars_ctx,
        )
        return SystemMessage(content=prompt_str)

    def _build_user_message(
        self,
        question: str,
        previous_sql: Optional[str],
        error_feedback: Optional[str],
    ) -> HumanMessage:
        if previous_sql and error_feedback:
            content = (
                f"Question: {question}\n\n"
                f"Previous SQL Attempt:\n{previous_sql}\n\n"
                f"Validation / Execution Error:\n{error_feedback}\n\n"
                "Please correct this query and provide only the corrected single SQLite SELECT statement."
            )
        else:
            content = question
        return HumanMessage(content=content)

    @traceable(name="SQLGeneration", run_type="chain")
    def generate(
        self,
        question: str,
        schema_context: str,
        history: Optional[List[Any]] = None,
        use_exemplars: bool = True,
        previous_sql: Optional[str] = None,
        error_feedback: Optional[str] = None,
    ) -> SQLGenerationResult:
        """
        Generate SQL from a natural-language question and schema context.

        Supports structured repair feedback via previous_sql and error_feedback.
        Returns SQLGenerationResult with is_generatable=False if the LLM
        signals it cannot produce a valid query from schema.
        """
        system = self._build_system_prompt(
            schema_context=schema_context,
            history=history,
            use_exemplars=use_exemplars,
            question=question,
            error_feedback=error_feedback,
        )
        human = self._build_user_message(
            question=question,
            previous_sql=previous_sql,
            error_feedback=error_feedback,
        )

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
        result = self._parse_generation_response(raw)

        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {
                    "question": question,
                    "schema_snippet": schema_context[:300],
                    "is_repair": bool(error_feedback),
                }
                run.outputs = {
                    "raw_output": raw,
                    "sql": result.sql,
                    "is_generatable": result.is_generatable,
                    "reason": result.reason,
                }
        except Exception:
            pass

        return result

    def repair(
        self,
        question: str,
        schema_context: str,
        previous_sql: str,
        error_feedback: str,
        history: Optional[List[Any]] = None,
    ) -> SQLGenerationResult:
        """Convenience method for structured SQL repair iteration."""
        return self.generate(
            question=question,
            schema_context=schema_context,
            history=history,
            use_exemplars=False,
            previous_sql=previous_sql,
            error_feedback=error_feedback,
        )


    def _parse_generation_response(self, raw: str) -> SQLGenerationResult:
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
