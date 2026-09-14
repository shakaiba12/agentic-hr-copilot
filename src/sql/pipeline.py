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
from src.core.state import SQLState
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import JudgeDecision
from src.guardrails.sql_guardrail import SQLGuardrail, SQLGuardrailResult
from src.sql.executor import ExecutionResult, SQLExecutor
from src.sql.generator import SQLGenerator
from src.sql.graph import SQLGraphBuilder, get_compiled_sql_graph
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
    """
    End-to-end SQL query generator, validator, and database executor.
    Acts as a thin compatibility façade around the LangGraph SQL Agent Subgraph.
    """

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

        self._builder = SQLGraphBuilder(
            settings=self.settings,
            schema_provider=self.schema_provider,
            generator=self.generator,
            guardrail=self.guardrail,
            executor=self.executor,
        )
        self.graph = self._builder.build_graph()
        self.workflow = self.graph.compile()

    @traceable(name="SQLPipeline", run_type="chain")
    def handle(
        self,
        query: str,
        history: Optional[List[Any]] = None,
    ) -> SQLPipelineResult:
        """Execute the SQL pipeline workflow for a validated DATA_QUERY."""
        try:
            initial_state: SQLState = {
                "query": query,
                "sanitized_query": query,
                "chat_history": history,
                "sql_retry_count": 0,
                "db_schema_context": self._schema_context,
            }

            final_state: SQLState = self.workflow.invoke(initial_state)
            result: Optional[SQLPipelineResult] = final_state.get("sql_result")

            if result is None:
                # Fallback reconstruction if sql_result is somehow unset
                sql_out = final_state.get("sql_output") or {}
                success = sql_out.get("success", False)
                result = SQLPipelineResult(
                    success=success,
                    query=query,
                    generated_sql=final_state.get("generated_sql"),
                    rows=final_state.get("sql_data") or [],
                    row_count=final_state.get("sql_row_count", 0),
                    message=final_state.get("candidate_answer", ""),
                    error=final_state.get("sql_error"),
                    judge_decision=final_state.get("judge_decision"),
                    retries_attempted=final_state.get("sql_retry_count", 0),
                )

            self._record_pipeline_output(
                query=query,
                sql=result.generated_sql or "",
                row_count=result.row_count,
                message=result.message,
                success=result.success,
                error=result.error,
                judge_decision=result.judge_decision,
            )
            return result

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

