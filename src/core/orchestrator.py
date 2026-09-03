"""
Master Orchestrator and Query Gate.
Single entrypoint for dispatching queries across Input Guardrails, QueryRouter,
Master Responses, RAG Pipeline, and SQL Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None

from src.core.config import Settings, get_settings
from src.core.observability import safe_trace_span
from src.guardrails.input_guardrail import InputGuardrailResult
from src.rag.pipeline import RAGPipeline, RAGPipelineResult
from src.rag.router import QueryRouter, RouteCategory, RouteDecision
from src.sql.pipeline import SQLPipeline, SQLPipelineResult


@dataclass
class OrchestratorResponse:
    """Structured response from the Master Orchestrator."""

    decision: RouteDecision
    source: str  # "master", "rag", "sql"
    response: str
    allowed: bool = True
    rag_result: Optional[RAGPipelineResult] = None
    sql_result: Optional[SQLPipelineResult] = None
    error: Optional[str] = None


class MasterOrchestrator:
    """
    Master Orchestrator Gate.
    Guarantees that every user query passes through deterministic guardrails
    and the QueryRouter before any RAG or SQL execution occurs.
    """

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        sql_pipeline: Optional[SQLPipeline] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.router = router or QueryRouter()
        self.rag_pipeline = rag_pipeline or RAGPipeline(self.settings)
        self.sql_pipeline = sql_pipeline or SQLPipeline(self.settings)

    @traceable(name="HR-Copilot-Request", run_type="chain")
    def process_query(
        self,
        query: str,
        history: Optional[Union[List[str], List[dict], List[Any]]] = None,
    ) -> OrchestratorResponse:
        """
        Main execution entrypoint:
        1. Evaluates Input Guardrails & QueryRouter deterministically.
        2. Blocks unsafe requests immediately at the Master boundary.
        3. Dispatches directly to Master for casual/general/out-of-scope inquiries.
        4. Dispatches to RAG Pipeline only for RAG_KNOWLEDGE queries.
        5. Dispatches to SQL Pipeline only for DATA_QUERY requests.
        """
        # Attach root trace inputs
        try:
            run = get_current_run_tree()
            if run:
                run.inputs = {"user_query": query}
        except Exception:
            pass

        # Step 1: Execute Guardrail and QueryRouter
        if hasattr(self.router, "guardrail") and self.router.guardrail is not None:
            guard_result = self.router.guardrail.check(query)
            try:
                decision = self.router.route(query, history=history, guard_result=guard_result)
            except TypeError:
                decision = self.router.route(query, history=history)
        else:
            decision = self.router.route(query, history=history)

        # Attach safe metadata to root trace
        try:
            run = get_current_run_tree()
            if run:
                run.metadata.update({
                    "environment": self.settings.APP_ENV,
                    "provider": self.settings.DEFAULT_PROVIDER,
                    "model": self.settings.DEFAULT_MODEL,
                    "route_category": decision.category.value,
                    "route_target": decision.target,
                    "allowed": decision.allowed,
                    "confidence": decision.confidence,
                })
        except Exception:
            pass

        # Step 2: Handle Blocked / Safety Violations / Invalid inputs (Fail-closed)
        if not decision.allowed:
            if decision.category == RouteCategory.INVALID:
                block_resp = f"ℹ️ {decision.reason}"
            else:
                block_resp = f"🛑 Request Blocked: {decision.reason}"

            self._record_root_output(decision, block_resp)
            return OrchestratorResponse(
                decision=decision,
                source="master",
                allowed=False,
                response=block_resp,
                error=decision.reason,
            )

        # Step 3: Handle Casual Greetings (Master Response)
        if decision.category == RouteCategory.CASUAL:
            with safe_trace_span(name="MasterPipeline", run_type="chain", inputs={"user_query": query, "category": "CASUAL"}):
                resp_text = (
                    "Hello! I am your Enterprise HR Intelligence Assistant. "
                    "I can help with company HR policies, benefits, leave, expenses, and employee information. "
                    "How can I help you today?"
                )
            self._record_root_output(decision, resp_text)
            return OrchestratorResponse(
                decision=decision,
                source="master",
                allowed=True,
                response=resp_text,
            )

        # Step 4: Handle Out-of-Scope Topics (Master Response)
        if decision.category == RouteCategory.OUT_OF_SCOPE:
            with safe_trace_span(name="MasterPipeline", run_type="chain", inputs={"user_query": query, "category": "OUT_OF_SCOPE"}):
                resp_text = (
                    "I'm designed to help with company HR policies, employee information, "
                    "benefits, leave, expenses, and related workplace questions. "
                    "Your query falls outside of approved enterprise documentation."
                )
            self._record_root_output(decision, resp_text)
            return OrchestratorResponse(
                decision=decision,
                source="master",
                allowed=True,
                response=resp_text,
            )

        # Step 5: Handle General Knowledge / Math / Definitions (Master Response)
        if decision.category == RouteCategory.GENERAL:
            with safe_trace_span(name="MasterPipeline", run_type="chain", inputs={"user_query": query, "category": "GENERAL"}):
                resp_text = (
                    f"This question was handled as a general inquiry: '{query}'. "
                    "I specialize in enterprise HR policies and company data."
                )
            self._record_root_output(decision, resp_text)
            return OrchestratorResponse(
                decision=decision,
                source="master",
                allowed=True,
                response=resp_text,
            )

        # Step 6: Dispatch to RAG Pipeline (Enterprise Policy & Knowledge)
        if decision.category == RouteCategory.RAG_KNOWLEDGE:
            rag_res: RAGPipelineResult = self.rag_pipeline.handle(query)
            self._record_root_output(decision, rag_res.response)
            return OrchestratorResponse(
                decision=decision,
                source="rag",
                allowed=True,
                response=rag_res.response,
                rag_result=rag_res,
            )

        # Step 7: Dispatch to SQL Pipeline (Operational Database & Records)
        if decision.category == RouteCategory.DATA_QUERY:
            sql_res: SQLPipelineResult = self.sql_pipeline.handle(query)
            self._record_root_output(decision, sql_res.message)
            return OrchestratorResponse(
                decision=decision,
                source="sql",
                allowed=True,
                response=sql_res.message,
                sql_result=sql_res,
                error=sql_res.error,
            )

        # Fallback Safe Master Response
        with safe_trace_span(name="MasterPipeline", run_type="chain", inputs={"user_query": query, "category": "FALLBACK"}):
            fallback_text = (
                "I could not determine the specific enterprise department for this question. "
                "Please ask about company HR policies, benefits, leave, or employee counts."
            )
        self._record_root_output(decision, fallback_text)
        return OrchestratorResponse(
            decision=decision,
            source="master",
            allowed=True,
            response=fallback_text,
        )

    @staticmethod
    def _record_root_output(decision: RouteDecision, response_text: str) -> None:
        try:
            run = get_current_run_tree()
            if run:
                run.outputs = {
                    "category": decision.category.value,
                    "target": decision.target,
                    "response": response_text,
                    "allowed": decision.allowed,
                }
        except Exception:
            pass
