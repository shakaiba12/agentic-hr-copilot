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
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import JudgeDecision, SQLJudgeInput
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
    judge_decision: Optional[JudgeDecision] = None
    retries_attempted: int = 0


class SQLPipeline:
    """End-to-end SQL query generator, validator, and database executor."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema_provider: Optional[SchemaProvider] = None,
        generator: Optional[SQLGenerator] = None,
        guardrail: Optional[SQLGuardrail] = None,
        executor: Optional[SQLExecutor] = None,
        judge: Optional[LLMJudge] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.schema_provider = schema_provider or SchemaProvider()
        self.generator = generator or SQLGenerator()
        self.guardrail = guardrail or SQLGuardrail(self.settings)
        self.executor = executor or SQLExecutor()
        self.judge = judge or LLMJudge()
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
            ) as schema_span:
                schema_context = self._schema_context
                if schema_span:
                    schema_span.end(outputs={"schema_length": len(schema_context), "tables": self.settings.ALLOWED_SQL_TABLES})

            # 2. Generate SQL from question and schema
            gen_result: SQLGenerationResult = self.generator.generate(
                query, schema_context, history=history
            )
            if not gen_result.is_generatable:
                msg = (
                    f"I can query employee salary and database records, but \"{gen_result.reason}\" "
                    "does not map clearly to a database field. What specific salary or employee data would you like to view?"
                )
                self._record_pipeline_output(query=query, sql="", row_count=0, message=msg, success=False, error=gen_result.reason)
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    message=msg,
                    error=gen_result.reason,
                )

            # 3. Validate SQL safety via guardrail
            validation: SQLGuardrailResult = self.guardrail.validate(gen_result.sql)
            if not validation.is_valid:
                msg = f"SQL Guardrail blocked query: {validation.notes}"
                self._record_pipeline_output(query=query, sql=gen_result.sql, row_count=0, message=msg, success=False, error=validation.notes)
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    generated_sql=gen_result.sql,
                    validation=validation,
                    message=msg,
                    error=validation.notes,
                )

            # 4. Execute safe read-only SQL
            exec_result: ExecutionResult = self.executor.execute(
                validation.normalized_sql
            )
            if not exec_result.success:
                msg = f"Database execution failed: {exec_result.error}"
                self._record_pipeline_output(query=query, sql=validation.normalized_sql, row_count=0, message=msg, success=False, error=exec_result.error)
                return SQLPipelineResult(
                    success=False,
                    query=query,
                    generated_sql=validation.normalized_sql,
                    validation=validation,
                    execution=exec_result,
                    message=msg,
                    error=exec_result.error,
                )

            # 5. Process and format results
            with safe_trace_span(
                name="SQL_ResultProcessing",
                run_type="chain",
                inputs={"row_count": exec_result.row_count, "truncated": exec_result.was_truncated},
            ) as res_span:
                result_message = f"Successfully retrieved {exec_result.row_count} rows from database."
                if res_span:
                    res_span.end(outputs={"message": result_message, "rows_count": exec_result.row_count})

            # 6. LLM-as-a-Judge Verification
            decision: Optional[JudgeDecision] = None
            if self.settings.ENABLE_LLM_JUDGE and exec_result.success:
                with safe_trace_span(
                    name="SQL_LLM_Judge",
                    run_type="chain",
                    inputs={"query": query, "sql": validation.normalized_sql, "rows_count": exec_result.row_count},
                ) as judge_span:
                    decision = self.judge.evaluate_sql(
                        SQLJudgeInput(
                            question=query,
                            sql=validation.normalized_sql,
                            sql_result=exec_result.rows,
                            answer=result_message,
                            history=history,
                        )
                    )
                    if judge_span:
                        judge_span.end(outputs={
                            "passed": decision.passed,
                            "score": decision.score,
                            "correctness": getattr(decision, "correctness", decision.score),
                            "verdict_status": decision.verdict_status,
                        })

            self._record_pipeline_output(
                query=query,
                sql=validation.normalized_sql,
                row_count=exec_result.row_count,
                message=result_message,
                success=True,
                judge_decision=decision,
            )

            return SQLPipelineResult(
                success=True,
                query=query,
                generated_sql=validation.normalized_sql,
                validation=validation,
                execution=exec_result,
                rows=exec_result.rows,
                row_count=exec_result.row_count,
                message=result_message,
                judge_decision=decision,
            )

        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                user_msg = "The LLM service is temporarily rate limited. Please try your query again in a few moments."
            else:
                user_msg = f"SQL Pipeline execution error: {exc}"
            self._record_pipeline_output(query=query, sql="", row_count=0, message=user_msg, success=False, error=err_str)
            return SQLPipelineResult(
                success=False,
                query=query,
                error=err_str,
                message=user_msg,
            )

    @staticmethod
    def _record_pipeline_output(
        query: str,
        sql: str,
        row_count: int,
        message: str,
        success: bool,
        error: Optional[str] = None,
        judge_decision: Optional[JudgeDecision] = None,
    ) -> None:
        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {"query": query}
                run.outputs = {
                    "sql": sql,
                    "row_count": row_count,
                    "message": message,
                    "success": success,
                    "error": error,
                    "judge_passed": judge_decision.passed if judge_decision else None,
                    "judge_score": judge_decision.score if judge_decision else None,
                }
        except Exception:
            pass
