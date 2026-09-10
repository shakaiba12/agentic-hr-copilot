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
from src.evaluation.schemas import JudgeDecision
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
    judge_decision: Optional[JudgeDecision] = None
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
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """
        Main execution entrypoint:
        1. Evaluates Input Guardrails & QueryRouter deterministically.
        2. Blocks unsafe requests immediately at the Master boundary.
        3. Dispatches directly to Master for casual/general/out-of-scope inquiries.
        4. Dispatches to RAG Pipeline only for RAG_KNOWLEDGE queries.
        5. Dispatches to SQL Pipeline only for DATA_QUERY requests.
        """
        import time
        import uuid
        from datetime import datetime, timezone

        start_time = time.perf_counter()
        active_req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        active_session_id = session_id or "default_session"
        start_ts = datetime.now(timezone.utc).isoformat()

        # Attach root trace inputs and initial metadata
        try:
            run = get_current_run_tree()
            if run:
                run.inputs = {"user_query": query}
                run.metadata.update({
                    "session_id": active_session_id,
                    "request_id": active_req_id,
                    "timestamp": start_ts,
                    "environment": self.settings.APP_ENV,
                    "provider": self.settings.DEFAULT_PROVIDER,
                    "model": self.settings.DEFAULT_MODEL,
                })
                if not hasattr(run, "tags") or not run.tags:
                    run.tags = ["hr-assistant"]
                elif "hr-assistant" not in run.tags:
                    run.tags.append("hr-assistant")
        except Exception:
            pass

        try:
            # Step 1: Execute Guardrail and QueryRouter
            if hasattr(self.router, "guardrail") and self.router.guardrail is not None:
                guard_result = self.router.guardrail.check(query)
                try:
                    decision = self.router.route(query, history=history, guard_result=guard_result)
                except TypeError:
                    decision = self.router.route(query, history=history)
            else:
                decision = self.router.route(query, history=history)

            pipeline_name = decision.target or "master"
            category_tag = decision.category.value.lower()

            # Attach route metadata and tags
            tags_to_add = [pipeline_name, category_tag]
            if not decision.allowed:
                tags_to_add.extend(["guardrail", "blocked"])
                if decision.category == RouteCategory.PROMPT_INJECTION:
                    tags_to_add.append("prompt-injection")
                elif decision.category == RouteCategory.DESTRUCTIVE_ACTION:
                    tags_to_add.append("destructive-action")

            self._update_root_meta(
                metadata={
                    "intent": decision.category.value,
                    "route": decision.target or "master",
                    "route_category": decision.category.value,
                    "route_target": decision.target,
                    "allowed": decision.allowed,
                    "confidence": decision.confidence,
                    "pipeline": pipeline_name,
                },
                tags=tags_to_add,
            )

            # Step 2: Handle Blocked / Safety Violations / Invalid inputs (Fail-closed)
            if not decision.allowed:
                if decision.category == RouteCategory.INVALID:
                    block_resp = f"ℹ️ {decision.reason}"
                else:
                    block_resp = f"🛑 Request Blocked: {decision.reason}"

                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                self._record_root_output(
                    decision=decision,
                    response_text=block_resp,
                    status="blocked",
                    latency_ms=elapsed_ms,
                    error=decision.reason,
                )
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
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                self._record_root_output(decision=decision, response_text=resp_text, status="success", latency_ms=elapsed_ms)
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
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                self._record_root_output(decision=decision, response_text=resp_text, status="success", latency_ms=elapsed_ms)
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
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                self._record_root_output(decision=decision, response_text=resp_text, status="success", latency_ms=elapsed_ms)
                return OrchestratorResponse(
                    decision=decision,
                    source="master",
                    allowed=True,
                    response=resp_text,
                )

            # Step 6: Dispatch to RAG Pipeline (Enterprise Policy & Knowledge)
            if decision.category == RouteCategory.RAG_KNOWLEDGE:
                self._update_root_meta(tags=["rag", "llm-judge"])
                rag_res: RAGPipelineResult = self.rag_pipeline.handle(query)
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                final_status = "success" if rag_res.success else "failure"
                self._record_root_output(
                    decision=decision,
                    response_text=rag_res.response,
                    status=final_status,
                    latency_ms=elapsed_ms,
                    judge_decision=rag_res.judge_decision,
                    error=rag_res.error,
                )
                return OrchestratorResponse(
                    decision=decision,
                    source="rag",
                    allowed=True,
                    response=rag_res.response,
                    rag_result=rag_res,
                    judge_decision=rag_res.judge_decision,
                    error=rag_res.error,
                )

            # Step 7: Dispatch to SQL Pipeline (Operational Database & Records)
            if decision.category == RouteCategory.DATA_QUERY:
                self._update_root_meta(tags=["sql", "llm-judge"])
                sql_res: SQLPipelineResult = self.sql_pipeline.handle(query)
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                final_status = "success" if sql_res.success else "failure"
                self._record_root_output(
                    decision=decision,
                    response_text=sql_res.message,
                    status=final_status,
                    latency_ms=elapsed_ms,
                    judge_decision=sql_res.judge_decision,
                    error=sql_res.error,
                )
                return OrchestratorResponse(
                    decision=decision,
                    source="sql",
                    allowed=True,
                    response=sql_res.message,
                    sql_result=sql_res,
                    judge_decision=sql_res.judge_decision,
                    error=sql_res.error,
                )

            # Fallback Safe Master Response
            with safe_trace_span(name="MasterPipeline", run_type="chain", inputs={"user_query": query, "category": "FALLBACK"}):
                fallback_text = (
                    "I could not determine the specific enterprise department for this question. "
                    "Please ask about company HR policies, benefits, leave, or employee counts."
                )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self._record_root_output(decision=decision, response_text=fallback_text, status="success", latency_ms=elapsed_ms)
            return OrchestratorResponse(
                decision=decision,
                source="master",
                allowed=True,
                response=fallback_text,
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = str(exc)
            fallback_err_resp = f"An unexpected system error occurred while processing your request: {err_msg}"
            fallback_decision = RouteDecision(
                category=RouteCategory.GENERAL,
                allowed=False,
                target=None,
                reason=err_msg,
                confidence=0.0,
            )
            self._update_root_meta(
                metadata={"status": "error", "error": err_msg, "total_latency_ms": elapsed_ms},
                tags=["error"],
            )
            self._record_root_output(
                decision=fallback_decision,
                response_text=fallback_err_resp,
                status="error",
                latency_ms=elapsed_ms,
                error=err_msg,
            )
            return OrchestratorResponse(
                decision=fallback_decision,
                source="master",
                allowed=False,
                response=fallback_err_resp,
                error=err_msg,
            )

    @staticmethod
    def _update_root_meta(metadata: Optional[dict] = None, tags: Optional[List[str]] = None) -> None:
        try:
            run = get_current_run_tree()
            if run:
                if metadata and hasattr(run, "metadata") and isinstance(run.metadata, dict):
                    run.metadata.update(metadata)
                if tags and hasattr(run, "tags") and isinstance(run.tags, list):
                    for t in tags:
                        if t not in run.tags:
                            run.tags.append(t)
        except Exception:
            pass

    @staticmethod
    def _record_root_output(
        decision: RouteDecision,
        response_text: str,
        status: str = "success",
        latency_ms: Optional[float] = None,
        judge_decision: Optional[JudgeDecision] = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            run = get_current_run_tree()
            if run:
                out_dict = {
                    "category": decision.category.value,
                    "target": decision.target,
                    "response": response_text,
                    "allowed": decision.allowed,
                    "status": status,
                }
                if latency_ms is not None:
                    out_dict["total_latency_ms"] = latency_ms
                if error:
                    out_dict["error"] = error
                if judge_decision:
                    out_dict["judge_passed"] = judge_decision.passed
                    out_dict["judge_score"] = judge_decision.score
                run.outputs = out_dict
                if hasattr(run, "metadata") and isinstance(run.metadata, dict):
                    run.metadata["status"] = status
                    if latency_ms is not None:
                        run.metadata["total_latency_ms"] = latency_ms
        except Exception:
            pass
