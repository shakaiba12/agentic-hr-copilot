"""
SQL Pipeline execution handler.
Encapsulates Schema Context -> SQL Generation -> SQL Guardrail -> Database Execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import Settings, get_settings
from src.core.observability import safe_trace_span
from src.guardrails.sql_guardrail import SQLGuardrail, SQLGuardrailResult
from src.sql.executor import ExecutionResult, SQLExecutor
from src.sql.generator import SQLGenerationResult, SQLGenerator
from src.sql.schema_provider import SchemaProvider


@dataclass
class SQLPipelineResult:
    """Structured result from SQL Pipeline execution."""

    success: bool
    query: str
    generated_sql: Optional[str] = None
    validation: Optional[SQLGuardrailResult] = None
    execution: Optional[ExecutionResult] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    message: str = ""


class SQLPipeline:
    """End-to-end SQL query generator, validator, and database executor."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema_provider: Optional[SchemaProvider] = None,
        generator: Optional[SQLGenerator] = None,
        guardrail: Optional[SQLGuardrail] = None,
        executor: Optional[SQLExecutor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.schema_provider = schema_provider or SchemaProvider()
        self.generator = generator or SQLGenerator()
        self.guardrail = guardrail or SQLGuardrail(self.settings)
        self.executor = executor or SQLExecutor()
        self._schema_context = self.schema_provider.get_full_schema()

    @traceable(name="SQLPipeline", run_type="chain")
    def handle(
        self,
        query: str,
        history: Optional[List[Any]] = None,
    ) -> SQLPipelineResult:
        """Execute the SQL pipeline for a validated DATA_QUERY."""
        try:
            # 1. Prepare Schema Context
            with safe_trace_span(
                name="SchemaContext",
                run_type="chain",
                inputs={"query": query},
                metadata={"tables_available": self.settings.ALLOWED_SQL_TABLES},
            ):
                schema_context = self._schema_context

            # 2. Generate SQL from question and schema
            gen_result: SQLGenerationResult = self.generator.generate(
                query, schema_context, history=history
            )
            if not gen_result.is_generatable:
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    message=(
                        f"I can query employee salary and database records, but \"{gen_result.reason}\" "
                        "does not map clearly to a database field. What specific salary or employee data would you like to view?"
                    ),
                    error=gen_result.reason,
                )

            # 3. Validate SQL safety via guardrail
            validation: SQLGuardrailResult = self.guardrail.validate(gen_result.sql)
            if not validation.is_valid:
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    generated_sql=gen_result.sql,
                    validation=validation,
                    message=f"SQL Guardrail blocked query: {validation.notes}",
                    error=validation.notes,
                )

            # 4. Execute safe read-only SQL
            exec_result: ExecutionResult = self.executor.execute(
                validation.normalized_sql
            )
            if not exec_result.success:
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    generated_sql=validation.normalized_sql,
                    validation=validation,
                    execution=exec_result,
                    message=f"Database execution failed: {exec_result.error}",
                    error=exec_result.error,
                )

            # 5. Process and format results
            with safe_trace_span(
                name="ResultProcessing",
                run_type="chain",
                inputs={"row_count": exec_result.row_count, "truncated": exec_result.was_truncated},
            ):
                result_message = f"Successfully retrieved {exec_result.row_count} rows from database."

            return SQLPipelineResult(
                success=True,
                query=query,
                generated_sql=validation.normalized_sql,
                validation=validation,
                execution=exec_result,
                rows=exec_result.rows,
                row_count=exec_result.row_count,
                message=result_message,
            )

        except Exception as exc:
            return SQLPipelineResult(
                success=False,
                query=query,
                error=str(exc),
                message=f"SQL Pipeline execution error: {exc}",
            )
