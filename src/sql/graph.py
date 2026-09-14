"""
LangGraph SQL Agent Subgraph.
Orchestrates: Schema Preparation -> SQL Generation -> SQL Validation -> SQL Execution -> SQL Repair Retry Loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import Settings, get_settings
from src.core.llm import get_llm
from src.core.observability import safe_trace_span
from src.core.state import SQLState
from src.guardrails.sql_guardrail import SQLGuardrail, SQLGuardrailResult
from src.sql.executor import ExecutionResult, SQLExecutor
from src.sql.generator import SQLGenerationResult, SQLGenerator
from src.sql.schema_provider import SchemaProvider

logger = logging.getLogger(__name__)


class SQLGraphBuilder:
    """Builder for the LangGraph SQL Agent Subgraph."""

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
        self.max_retries = getattr(self.settings, "MAX_SQL_RETRIES", 2)
        self._schema_context = self.schema_provider.get_full_schema()

    def schema_preparation_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 1: Extract and prepare database schema context."""
        query = state.get("sanitized_query") or state.get("query", "")
        with safe_trace_span(
            name="SchemaContext",
            run_type="chain",
            inputs={"query": query},
            metadata={"tables_available": self.settings.ALLOWED_SQL_TABLES},
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
                f"I can query employee salary and database records, but \"{reason}\" "
                "does not map clearly to a database field. What specific salary or employee data would you like to view?"
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
        """Node 3: Deterministically validate SQL safety, tables, and keywords."""
        sql = state.get("generated_sql", "")
        if not sql:
            return {
                "is_sql_valid": False,
                "sql_validation_notes": state.get("sql_error") or "No SQL generated.",
            }

        validation: SQLGuardrailResult = self.guardrail.validate(sql)
        updates: Dict[str, Any] = {
            "is_sql_valid": validation.is_valid,
            "sql_validation_notes": validation.notes,
        }

        if validation.is_valid:
            if validation.normalized_sql:
                updates["generated_sql"] = validation.normalized_sql
        else:
            updates["sql_error"] = validation.notes

        return updates

    def sql_execution_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 4: Execute validated SQL read-only query against SQLite."""
        sql = state.get("generated_sql", "")
        exec_result: ExecutionResult = self.executor.execute(sql)

        if not exec_result.success:
            return {
                "sql_data": [],
                "sql_row_count": 0,
                "sql_error": exec_result.error,
                "errors": [exec_result.error] if exec_result.error else [],
            }

        return {
            "sql_data": exec_result.rows,
            "sql_row_count": exec_result.row_count,
            "sql_error": None,
        }

    def sql_repair_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 5: Increment retry count, build error feedback, and re-generate query."""
        current_retries = state.get("sql_retry_count", 0) + 1
        query = state.get("sanitized_query") or state.get("query", "")
        previous_sql = state.get("generated_sql", "")
        error_msg = state.get("sql_error") or state.get("sql_validation_notes") or "SQL generation failed."
        schema_context = state.get("db_schema_context") or self._schema_context
        history = state.get("chat_history")

        logger.info(
            f"SQL Subgraph Repair Attempt {current_retries}/{self.max_retries}. Error: {error_msg}"
        )

        repair_query = (
            f"{query}\n\n"
            f"[Correction Request]: Previous SQL was '{previous_sql}'. Error: {error_msg}. "
            "Please fix this error and generate a single valid SELECT query using only allowed tables and columns."
        )

        gen_result: SQLGenerationResult = self.generator.generate(
            repair_query, schema_context, history=history
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
            return f"SQL Guardrail blocked query: {notes}"

        if sql_error:
            return f"Database execution failed: {sql_error}"

        return state.get("sql_validation_notes") or "SQL generation or execution failed."

    def format_sql_result_node(self, state: SQLState) -> Dict[str, Any]:
        """Node 6: Finalize structured SQL pipeline output and candidate message."""
        from src.sql.pipeline import SQLPipelineResult

        query = state.get("sanitized_query") or state.get("query", "")
        generated_sql = state.get("generated_sql")
        is_valid = bool(state.get("is_sql_valid", False))
        sql_error = state.get("sql_error")
        rows = state.get("sql_data") or []
        row_count = state.get("sql_row_count", len(rows))
        retries = state.get("sql_retry_count", 0)

        success = is_valid and sql_error is None
        msg = self._build_candidate_message(state, is_valid, sql_error, row_count)

        if success and rows:
            try:
                llm = get_llm(temperature=0.0)
                prompt = (
                    f"You are PeopleQuery AI, an Enterprise HR Intelligence Copilot.\n"
                    f"Provide a direct, concise, and professional natural language answer to the user's question based strictly on the SQL query results below. Do NOT mention SQL query details or table names unless asked.\n\n"
                    f"User Question: \"{query}\"\n"
                    f"Query Results ({row_count} rows):\n{rows[:25]}\n\n"
                    f"Direct Answer:"
                )
                res = llm.invoke(prompt)
                raw_c = res.content if hasattr(res, "content") else str(res)
                if isinstance(raw_c, list):
                    text = "".join(
                        chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
                        for chunk in raw_c
                    )
                else:
                    text = str(raw_c)
                if text.strip():
                    msg = text.strip()
            except Exception as e:
                logger.debug("SQL answer synthesis fallback: %s", e)
        elif success and not rows:
            msg = "No matching employee or workforce records were found in the database."

        # Validation object reconstruction
        validation_obj: Optional[SQLGuardrailResult] = None
        if generated_sql:
            validation_obj = SQLGuardrailResult(
                is_valid=is_valid,
                normalized_sql=generated_sql,
                notes=state.get("sql_validation_notes", ""),
            )

        # Execution object reconstruction
        exec_obj: Optional[ExecutionResult] = None
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
) -> StateGraph:
    """Factory creating uncompiled StateGraph for the SQL Agent."""
    builder = SQLGraphBuilder(
        settings=settings,
        schema_provider=schema_provider,
        generator=generator,
        guardrail=guardrail,
        executor=executor,
    )
    return builder.build_graph()


def get_compiled_sql_graph(
    settings: Optional[Settings] = None,
    schema_provider: Optional[SchemaProvider] = None,
    generator: Optional[SQLGenerator] = None,
    guardrail: Optional[SQLGuardrail] = None,
    executor: Optional[SQLExecutor] = None,
) -> Any:
    """Factory creating compiled executable LangGraph workflow for the SQL Agent."""
    graph = create_sql_graph(
        settings=settings,
        schema_provider=schema_provider,
        generator=generator,
        guardrail=guardrail,
        executor=executor,
    )
    return graph.compile()
