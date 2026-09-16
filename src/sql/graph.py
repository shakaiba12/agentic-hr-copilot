"""
LangGraph SQL Agent Subgraph.
Orchestrates: Schema Context -> NL2SQL Generation -> Dual Validation (AST + Dry-Run) -> Execution -> Bounded Repair (<= 2).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from src.core.config import Settings, get_settings
from src.core.observability import safe_trace_span
from src.core.state import SQLState
from src.guardrails.sql_guardrail import SQLGuardrail, SQLGuardrailResult
from src.sql.executor import ExecutionResult, SQLExecutor
from src.sql.formatter import SQLResultPresenter, render_markdown_table
from src.sql.generator import SQLGenerationResult, SQLGenerator
from src.sql.schema_provider import SchemaProvider

logger = logging.getLogger(__name__)


def format_markdown_table(rows: List[Dict[str, Any]], max_rows: int = 25) -> str:
    """Format a list of dictionary rows into a clean Markdown table."""
    return render_markdown_table(rows, max_rows=max_rows)


class SQLGraphBuilder:
    """Builder for the LangGraph SQL Agent Subgraph."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema_provider: Optional[SchemaProvider] = None,
        generator: Optional[SQLGenerator] = None,
        guardrail: Optional[SQLGuardrail] = None,
        executor: Optional[SQLExecutor] = None,
        presenter: Optional[SQLResultPresenter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.schema_provider = schema_provider or SchemaProvider()
        self.generator = generator or SQLGenerator()
        self.guardrail = guardrail or SQLGuardrail(self.settings, schema_provider=self.schema_provider)
        self.executor = executor or SQLExecutor()
        self.presenter = presenter or SQLResultPresenter()
        self.max_retries = getattr(self.settings, "MAX_SQL_RETRIES", 2)
        self._schema_context = self.schema_provider.get_full_schema()

    def schema_preparation_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 1: Extract and prepare enriched database schema context."""
        query = state.get("sanitized_query") or state.get("query", "")
        with safe_trace_span(
            name="SchemaContext",
            run_type="chain",
            inputs={"query": query},
            metadata={"tables_available": self.schema_provider.get_table_names()},
        ) as schema_span:
            schema_context = self._schema_context
            if schema_span:
                schema_span.end(outputs={"schema_length": len(schema_context)})
        return {"db_schema_context": schema_context}

    def sql_generation_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 2: Generate natural language SQL SELECT from schema."""
        query = state.get("sanitized_query") or state.get("query", "")
        schema_context = state.get("db_schema_context") or self._schema_context
        history = state.get("chat_history")

        gen_result: SQLGenerationResult = self.generator.generate(
            query, schema_context, history=history
        )

        if not gen_result.is_generatable:
            reason = gen_result.reason
            msg = (
                f"I can query employee salary, benefits, leave, and department records, but \"{reason}\" "
                "does not map to available HR database fields. What specific employee or workforce data would you like to view?"
            )
            return {
                "generated_sql": "",
                "is_sql_valid": False,
                "sql_error": reason,
                "sql_validation_notes": reason,
                "candidate_answer": msg,
                "errors": [reason],
            }

        return {
            "generated_sql": gen_result.sql,
            "is_sql_valid": None,
            "sql_error": None,
        }

    def sql_validation_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 3: Dual-stage validation: Stage 1 AST Safety + Stage 2 SQLite Dry-Run."""
        sql = state.get("generated_sql", "")
        if not sql:
            return {
                "is_sql_valid": False,
                "sql_validation_notes": state.get("sql_error") or "No SQL generated.",
            }

        # Stage 1: Deterministic AST Safety
        validation: SQLGuardrailResult = self.guardrail.validate(sql)
        if not validation.is_valid:
            return {
                "is_sql_valid": False,
                "sql_validation_notes": validation.notes,
                "sql_error": validation.notes,
                "sql_validation_result": validation,
            }

        normalized_sql = validation.normalized_sql or sql

        # Stage 2: SQLite Dry-Run (EXPLAIN QUERY PLAN)
        dry_run_valid, dry_run_error = self.executor.dry_run(normalized_sql)
        if not dry_run_valid:
            return {
                "is_sql_valid": False,
                "generated_sql": normalized_sql,
                "sql_validation_notes": f"Dry-run compilation error: {dry_run_error}",
                "sql_error": dry_run_error,
                "sql_validation_result": validation,
            }

        return {
            "is_sql_valid": True,
            "generated_sql": normalized_sql,
            "sql_validation_notes": "Passed AST safety and SQLite dry-run validation.",
            "sql_error": None,
            "sql_validation_result": validation,
        }

    def sql_execution_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 4: Execute validated SQL read-only query against SQLite."""
        sql = state.get("generated_sql", "")
        exec_result: ExecutionResult = self.executor.execute(sql)

        if not exec_result.success:
            return {
                "sql_data": [],
                "sql_row_count": 0,
                "sql_error": exec_result.error,
                "sql_execution_result": exec_result,
                "errors": [exec_result.error] if exec_result.error else [],
            }

        return {
            "sql_data": exec_result.rows,
            "sql_row_count": exec_result.row_count,
            "sql_error": None,
            "sql_execution_result": exec_result,
        }

    def sql_repair_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 5: Increment retry count and re-generate using structured error feedback."""
        current_retries = state.get("sql_retry_count", 0) + 1
        query = state.get("sanitized_query") or state.get("query", "")
        previous_sql = state.get("generated_sql", "")
        error_msg = state.get("sql_error") or state.get("sql_validation_notes") or "SQL generation failed."
        schema_context = state.get("db_schema_context") or self._schema_context
        history = state.get("chat_history")

        logger.info(
            f"SQL Subgraph Repair Attempt {current_retries}/{self.max_retries}. Error: {error_msg}"
        )

        gen_result: SQLGenerationResult = self.generator.repair(
            question=query,
            schema_context=schema_context,
            previous_sql=previous_sql,
            error_feedback=error_msg,
            history=history,
        )

        if not gen_result.is_generatable:
            return {
                "sql_retry_count": current_retries,
                "generated_sql": "",
                "is_sql_valid": False,
                "sql_error": gen_result.reason,
                "sql_validation_notes": gen_result.reason,
            }

        return {
            "sql_retry_count": current_retries,
            "generated_sql": gen_result.sql,
            "is_sql_valid": None,
            "sql_error": None,
        }

    @staticmethod
    def _build_candidate_message(
        state: SQLState,
        is_valid: bool,
        sql_error: Optional[str],
        row_count: int,
    ) -> str:
        """Resolve user-facing candidate answer message from execution and validation state."""
        if is_valid and sql_error is None:
            return f"Successfully retrieved {row_count} rows from database."

        candidate_answer = state.get("candidate_answer")
        if candidate_answer and not candidate_answer.startswith("Successfully"):
            return candidate_answer

        if not is_valid:
            notes = state.get("sql_validation_notes") or sql_error or "SQL failed validation."
            return f"SQL safety guardrail: {notes}"

        if sql_error:
            return f"Database execution error: {sql_error}"

        return state.get("sql_validation_notes") or "SQL generation or execution failed."

    def format_sql_result_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 6: Finalize structured SQL pipeline output, Markdown table, and candidate message."""
        from src.sql.pipeline import SQLPipelineResult

        query = state.get("sanitized_query") or state.get("query", "")
        generated_sql = state.get("generated_sql")
        is_valid = bool(state.get("is_sql_valid", False))
        sql_error = state.get("sql_error")
        rows = state.get("sql_data") or []
        row_count = state.get("sql_row_count", len(rows))
        retries = state.get("sql_retry_count", 0)

        success = is_valid and sql_error is None
        if success:
            msg = self.presenter.present(
                question=query,
                rows=rows,
                row_count=row_count,
                generated_sql=generated_sql,
                is_valid=is_valid,
                sql_error=sql_error,
            )
        else:
            msg = self._build_candidate_message(state, is_valid, sql_error, row_count)

        validation_obj: Optional[SQLGuardrailResult] = state.get("sql_validation_result")
        if validation_obj is None and generated_sql:
            validation_obj = SQLGuardrailResult(
                is_valid=is_valid,
                normalized_sql=generated_sql,
                notes=state.get("sql_validation_notes", ""),
            )

        exec_obj: Optional[ExecutionResult] = state.get("sql_execution_result")
        if exec_obj is None:
            if success:
                exec_obj = ExecutionResult(
                    success=True,
                    rows=rows,
                    row_count=row_count,
                )
            elif sql_error:
                exec_obj = ExecutionResult(
                    success=False,
                    error=sql_error,
                )

        pipeline_result = SQLPipelineResult(
            success=success,
            query=query,
            generated_sql=generated_sql if is_valid else None,
            validation=validation_obj,
            execution=exec_obj,
            rows=rows,
            row_count=row_count,
            error=sql_error,
            message=msg,
            judge_decision=None,
            retries_attempted=retries,
        )

        return {
            "sql_output": {
                "success": success,
                "generated_sql": generated_sql,
                "rows": rows,
                "row_count": row_count,
                "message": msg,
                "error": sql_error,
            },
            "sql_result": pipeline_result,
            "generated_sql": generated_sql if is_valid else None,
            "sql_data": rows,
            "sql_row_count": row_count,
            "is_sql_valid": is_valid,
            "candidate_answer": msg,
        }

    def _route_after_sql_validation(self, state: SQLState) -> str:
        """Pure routing function after validation check."""
        is_valid = state.get("is_sql_valid", False)
        retries = state.get("sql_retry_count", 0)

        if is_valid:
            return "sql_execution"
        if retries < self.max_retries and state.get("generated_sql"):
            return "sql_repair"
        return "format_sql_result"

    def _route_after_sql_execution(self, state: SQLState) -> str:
        """Pure routing function after database execution check."""
        error = state.get("sql_error")
        retries = state.get("sql_retry_count", 0)

        if error and retries < self.max_retries:
            return "sql_repair"
        return "format_sql_result"

    def build_graph(self) -> StateGraph:
        """Construct the compiled LangGraph SQL Agent Subgraph."""
        builder = StateGraph(SQLState)

        builder.add_node("schema_preparation", self.schema_preparation_node)
        builder.add_node("sql_generation", self.sql_generation_node)
        builder.add_node("sql_validation", self.sql_validation_node)
        builder.add_node("sql_execution", self.sql_execution_node)
        builder.add_node("sql_repair", self.sql_repair_node)
        builder.add_node("format_sql_result", self.format_sql_result_node)

        builder.add_edge(START, "schema_preparation")
        builder.add_edge("schema_preparation", "sql_generation")
        builder.add_edge("sql_generation", "sql_validation")

        builder.add_conditional_edges(
            "sql_validation",
            self._route_after_sql_validation,
            {
                "sql_execution": "sql_execution",
                "sql_repair": "sql_repair",
                "format_sql_result": "format_sql_result",
            },
        )

        builder.add_conditional_edges(
            "sql_execution",
            self._route_after_sql_execution,
            {
                "format_sql_result": "format_sql_result",
                "sql_repair": "sql_repair",
            },
        )

        builder.add_edge("sql_repair", "sql_validation")
        builder.add_edge("format_sql_result", END)

        return builder


def create_sql_graph(
    settings: Optional[Settings] = None,
    schema_provider: Optional[SchemaProvider] = None,
    generator: Optional[SQLGenerator] = None,
    guardrail: Optional[SQLGuardrail] = None,
    executor: Optional[SQLExecutor] = None,
    presenter: Optional[SQLResultPresenter] = None,
) -> StateGraph:
    """Factory creating uncompiled StateGraph for the SQL Agent."""
    builder = SQLGraphBuilder(
        settings=settings,
        schema_provider=schema_provider,
        generator=generator,
        guardrail=guardrail,
        executor=executor,
        presenter=presenter,
    )
    return builder.build_graph()


def get_compiled_sql_graph(
    settings: Optional[Settings] = None,
    schema_provider: Optional[SchemaProvider] = None,
    generator: Optional[SQLGenerator] = None,
    guardrail: Optional[SQLGuardrail] = None,
    executor: Optional[SQLExecutor] = None,
    presenter: Optional[SQLResultPresenter] = None,
) -> Any:
    """Factory creating compiled executable LangGraph workflow for the SQL Agent."""
    graph = create_sql_graph(
        settings=settings,
        schema_provider=schema_provider,
        generator=generator,
        guardrail=guardrail,
        executor=executor,
        presenter=presenter,
    )
    return graph.compile()
