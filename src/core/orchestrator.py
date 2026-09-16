"""
Master Orchestrator and Query Gate built with LangGraph.
Coordinates Input Guardrails, QueryRouter, SQL/RAG Pipelines, Hybrid Synthesis,
LLM Judge, Regeneration, and Output Guardrails in a streamlined StateGraph.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from langgraph.graph import END, START, StateGraph

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import Settings, get_settings
from src.core.observability import (
    build_trace_tags,
    record_root_trace,
    update_trace_request,
)
from src.core.router import QueryRouter, RouteCategory, RouteDecision
from src.core.routing import (
    CATEGORY_TO_INTENT,
    DEFAULT_FALLBACK_ANSWER,
    INTENT_FALLBACK_ROUTES,
    INTENT_ROUTES,
    STATIC_CATEGORY_RESPONSES,
)
from src.core.state import AgentState, IntentType, SQLState
from src.core.synthesis import ConversationalResponseHandler, HybridSynthesizer
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import (
    HybridJudgeInput,
    JudgeDecision,
    RAGJudgeInput,
    SQLJudgeInput,
    VerdictStatus,
)

from src.guardrails.input_guardrail import InputGuardrail, InputGuardrailResult
from src.guardrails.output_guardrail import OutputGuardrail, OutputGuardrailResult
from src.rag.graph import get_compiled_rag_graph
from src.rag.pipeline import RAGPipeline, RAGPipelineResult
from src.rag.state import RAGState
from src.rag.utils import normalize_rag_sources
from src.sql.graph import get_compiled_sql_graph
from src.sql.pipeline import SQLPipeline, SQLPipelineResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    """Structured response object returned by MasterOrchestrator."""
    decision: RouteDecision
    source: str
    response: str
    allowed: bool = True
    rag_result: Optional[RAGPipelineResult] = None
    sql_result: Optional[SQLPipelineResult] = None
    judge_decision: Optional[JudgeDecision] = None
    error: Optional[str] = None


class MasterOrchestrator:
    """
    Thin Orchestration Layer coordinating guardrails, router, subgraphs,
    hybrid synthesis, and evaluation judge.
    """

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        sql_pipeline: Optional[SQLPipeline] = None,
        input_guardrail: Optional[InputGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
        judge: Optional[LLMJudge] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.router = router or QueryRouter()
        self.rag_pipeline = rag_pipeline or RAGPipeline(self.settings)
        self.sql_pipeline = sql_pipeline or SQLPipeline(self.settings)
        self.input_guardrail = (
            input_guardrail
            or getattr(self.router, "guardrail", None)
            or InputGuardrail(self.settings)
        )
        self.output_guardrail = output_guardrail or OutputGuardrail(self.settings)
        self.judge = judge or LLMJudge()

        # Dedicated synthesis and conversational delegates
        self.hybrid_synthesizer = HybridSynthesizer()
        self.conversational_handler = ConversationalResponseHandler()

        # Compiled subgraphs composed into master orchestrator
        self.sql_workflow = getattr(self.sql_pipeline, "workflow", None) or get_compiled_sql_graph(settings=self.settings)
        self.rag_workflow = getattr(self.rag_pipeline, "workflow", None) or get_compiled_rag_graph(settings=self.settings)

        # Build and compile master state graph
        self.graph = self._build_graph()
        self.workflow = self.graph.compile()

    def input_guardrail_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Deterministic safety validation and sanitization."""
        raw_query = state.get("query", "")
        guard_res: InputGuardrailResult = self.input_guardrail.check(raw_query)

        updates: Dict[str, Any] = {
            "is_input_safe": guard_res.is_safe,
            "sanitized_query": guard_res.sanitized_query,
            "input_rejection_reason": guard_res.rejection_reason,
        }

        if not guard_res.is_safe:
            decision = self.router.route(raw_query, guard_result=guard_res)
            block_msg = (
                f" {decision.reason}"
                if decision.category == RouteCategory.INVALID or not raw_query.strip()
                else f" Request Blocked: {decision.reason}"
            )
            updates.update({
                "route_decision": decision,
                "routing_output": {
                    "category": decision.category.value,
                    "allowed": False,
                    "target": None,
                    "reason": decision.reason,
                    "confidence": 1.0,
                },
                "candidate_answer": block_msg,
                "source": "master",
                "intent": IntentType.UNKNOWN,
            })

        return updates

    def intent_router_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Semantic and heuristic query intent classification."""
        if state.get("is_input_safe") is False:
            return {}

        query = state.get("sanitized_query") or state.get("query", "")
        history = state.get("chat_history")
        decision: RouteDecision = self.router.route(query, history=history)

        updates: Dict[str, Any] = {
            "route_decision": decision,
            "routing_output": {
                "category": decision.category.value,
                "allowed": decision.allowed,
                "target": decision.target,
                "reason": decision.reason,
                "confidence": decision.confidence,
            },
            "intent_reasoning": decision.reason,
            "is_input_safe": decision.allowed,
        }

        if not decision.allowed:
            block_msg = (
                f"{decision.reason}"
                if decision.category == RouteCategory.INVALID
                else f" Request Blocked: {decision.reason}"
            )
            updates.update({
                "candidate_answer": block_msg,
                "source": "master",
                "intent": IntentType.UNKNOWN,
            })
            return updates

        explicit_intent = state.get("intent")
        if explicit_intent is not None and explicit_intent != IntentType.UNKNOWN:
            updates["intent"] = explicit_intent
            if explicit_intent == IntentType.HYBRID:
                updates["source"] = "hybrid"
            return updates

        intent, source = CATEGORY_TO_INTENT.get(
            decision.category, (IntentType.UNKNOWN, "master")
        )
        updates["intent"] = intent
        updates["source"] = source
        return updates

    def sql_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3A: Directly executes compiled SQL Agent LangGraph Subgraph."""
        query = state.get("sanitized_query") or state.get("query", "")

        try:
            schema_context = (
                getattr(self.sql_pipeline, "_schema_context", None)
                or getattr(getattr(self.sql_pipeline, "schema_provider", None), "get_full_schema", lambda: "")()
            )
            sql_input: SQLState = {
                "query": query,
                "sanitized_query": query,
                "chat_history": state.get("chat_history"),
                "sql_retry_count": 0,
                "db_schema_context": schema_context,
            }

            final_sql_state: SQLState = self.sql_workflow.invoke(sql_input)
            sql_res: Optional[SQLPipelineResult] = final_sql_state.get("sql_result")
            sql_out = final_sql_state.get("sql_output") or {}

            updates: Dict[str, Any] = {
                "sql_result": sql_res,
                "sql_output": sql_out,
                "generated_sql": final_sql_state.get("generated_sql"),
                "sql_data": final_sql_state.get("sql_data", []),
                "sql_row_count": final_sql_state.get("sql_row_count", 0),
                "is_sql_valid": final_sql_state.get("is_sql_valid", False),
            }

            if state.get("intent") == IntentType.SQL_ONLY:
                updates["candidate_answer"] = final_sql_state.get("candidate_answer", "")
                updates["source"] = "sql"

            if final_sql_state.get("errors"):
                updates["errors"] = final_sql_state.get("errors")
            elif sql_res and not sql_res.success and sql_res.error:
                updates["errors"] = [sql_res.error]

            return updates
        except Exception as exc:
            logger.exception("SQL agent node error: %s", exc)
            return {
                "errors": [str(exc)],
                "candidate_answer": "An error occurred while processing the database query. Please try again later.",
                "sql_output": {"success": False, "error": str(exc)},
            }

    def rag_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3B: Directly executes compiled RAG Agent LangGraph Subgraph."""
        query = state.get("sanitized_query") or state.get("query", "")

        try:
            rag_input: RAGState = {
                "query": query,
                "sanitized_query": query,
                "chat_history": state.get("chat_history"),
            }

            final_rag_state: RAGState = self.rag_workflow.invoke(rag_input)
            rag_res: Optional[RAGPipelineResult] = final_rag_state.get("rag_result")
            rag_out = final_rag_state.get("rag_output") or {}
            retrieved_chunks = final_rag_state.get("retrieved_chunks", [])
            citations = final_rag_state.get("citations", [])

            updates: Dict[str, Any] = {
                "rag_result": rag_res,
                "rag_output": rag_out,
                "retrieved_chunks": retrieved_chunks,
                "citations": citations,
            }

            if state.get("intent") == IntentType.RAG_ONLY:
                updates["candidate_answer"] = final_rag_state.get("candidate_answer", "")
                updates["source"] = "rag"

            if final_rag_state.get("errors"):
                updates["errors"] = final_rag_state.get("errors")
            elif rag_res and not rag_res.success and rag_res.error:
                updates["errors"] = [rag_res.error]

            return updates

        except Exception as exc:
            logger.exception("RAG agent node error: %s", exc)
            return {
                "errors": [str(exc)],
                "candidate_answer": "An error occurred while searching enterprise documentation. Please try again later.",
                "rag_output": {"success": False, "error": str(exc)},
            }

    def answer_synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Synthesizes responses for HYBRID, CASUAL, and GENERAL queries."""
        intent = state.get("intent")
        query = state.get("sanitized_query") or state.get("query", "")
        decision: Optional[RouteDecision] = state.get("route_decision")

        # Non-hybrid candidate answer already populated by specialist agent
        if state.get("candidate_answer") and intent not in (IntentType.HYBRID, None):
            return {"candidate_answer": state["candidate_answer"]}

        # Parallel fan-in synthesis for HYBRID queries
        if intent == IntentType.HYBRID:
            sql_out = state.get("sql_output") or {}
            rag_out = state.get("rag_output") or {}
            synth_text = self.hybrid_synthesizer.synthesize(
                query=query,
                sql_data=state.get("sql_data") or [],
                retrieved_chunks=state.get("retrieved_chunks") or [],
                sql_success=bool(sql_out.get("success", True)),
                rag_success=bool(rag_out.get("success", True)),
                sql_msg=sql_out.get("message", ""),
                rag_msg=rag_out.get("response", ""),
            )
            return {"candidate_answer": synth_text, "source": "hybrid"}

        # Conversational, Casual, or Fallback handling
        category = decision.category if decision else RouteCategory.CASUAL
        response_text = self.conversational_handler.respond(query=query, category=category)
        return {"candidate_answer": response_text, "source": "master"}

    def _evaluate_hybrid_judge(self, query: str, candidate: str, state: AgentState) -> JudgeDecision:
        return self.judge.evaluate_hybrid(
            HybridJudgeInput(
                question=query,
                sql=state.get("generated_sql", "") or "",
                sql_result=state.get("sql_data", []) or [],
                retrieved_docs=state.get("retrieved_chunks", []) or [],
                answer=candidate,
                history=state.get("chat_history"),
            )
        )

    def _evaluate_rag_judge(self, query: str, candidate: str, state: AgentState) -> JudgeDecision:
        return self.judge.evaluate_rag(
            RAGJudgeInput(
                question=query,
                answer=candidate,
                retrieved_docs=state.get("retrieved_chunks", []) or [],
                history=state.get("chat_history"),
            )
        )

    def _evaluate_sql_judge(self, query: str, candidate: str, state: AgentState) -> JudgeDecision:
        return self.judge.evaluate_sql(
            SQLJudgeInput(
                question=query,
                sql=state.get("generated_sql", "") or "",
                sql_result=state.get("sql_data", []) or [],
                answer=candidate,
                history=state.get("chat_history"),
            )
        )

    def llm_judge_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Central LLM Judge evaluating answer quality and grounding."""
        # Guard clause: skip judging for unsafe, casual, or judge-disabled queries
        if (
            state.get("is_input_safe") is False
            or state.get("intent") in (IntentType.CASUAL, IntentType.UNKNOWN)
            or not getattr(self.settings, "ENABLE_LLM_JUDGE", True)
        ):
            decision = JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.SKIPPED,
                reason="Skipped for conversational / blocked query or judge disabled.",
            )
            return {
                "judge_decision": decision,
                "judge_output": {
                    "decision": "PASS",
                    "score": 1.0,
                    "passed": True,
                    "issues": [],
                    "feedback": None,
                },
            }

        intent = state.get("intent")
        query = state.get("sanitized_query") or state.get("query", "")
        candidate = state.get("candidate_answer", "")

        judge_dispatch = {
            IntentType.HYBRID: self._evaluate_hybrid_judge,
            IntentType.RAG_ONLY: self._evaluate_rag_judge,
            IntentType.SQL_ONLY: self._evaluate_sql_judge,
        }

        try:
            evaluator = judge_dispatch.get(intent)
            if evaluator:
                decision = evaluator(query, candidate, state)
            else:
                decision = JudgeDecision(
                    passed=True,
                    score=1.0,
                    verdict_status=VerdictStatus.SKIPPED,
                    reason="Evaluation not required for intent.",
                )
        except Exception as exc:
            logger.exception("LLM Judge node error: %s", exc)
            decision = JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.SKIPPED,
                reason=f"Judge evaluation skipped: {exc}",
            )

        return {
            "judge_decision": decision,
            "judge_output": {
                "decision": "PASS" if decision.passed else "FAIL",
                "score": decision.score,
                "passed": decision.passed,
                "issues": decision.issues,
                "feedback": decision.reason,
            },
        }

    def _regenerate_rag(
        self,
        query: str,
        candidate: str,
        feedback: str,
        history: Optional[List[Any]],
        state: AgentState,
    ) -> Optional[str]:
        generator = getattr(self.rag_pipeline, "generator", None)
        context_builder = getattr(self.rag_pipeline, "context_builder", None)
        if not (generator and hasattr(generator, "regenerate") and context_builder):
            return None

        chunks = state.get("retrieved_chunks") or []
        formatted_ctx = context_builder.build_context(chunks)
        if not formatted_ctx:
            return None

        regen_res = generator.regenerate(
            query=query,
            formatted_context=formatted_ctx,
            previous_answer=candidate,
            judge_feedback=feedback,
            history=history,
        )
        return regen_res.answer if hasattr(regen_res, "answer") else str(regen_res)

    def _regenerate_hybrid(
        self,
        query: str,
        candidate: str,
        feedback: str,
        history: Optional[List[Any]],
        state: AgentState,
    ) -> Optional[str]:
        sql_out = state.get("sql_output") or {}
        rag_out = state.get("rag_output") or {}
        return self.hybrid_synthesizer.regenerate(
            query=query,
            sql_data=state.get("sql_data") or [],
            retrieved_chunks=state.get("retrieved_chunks") or [],
            sql_success=bool(sql_out.get("success", True)),
            rag_success=bool(rag_out.get("success", True)),
            sql_msg=sql_out.get("message", ""),
            rag_msg=rag_out.get("response", ""),
            judge_feedback=feedback,
        )

    def regeneration_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 6: Bounded regeneration node re-invoking generator/synthesizer with critique."""
        retries = state.get("retry_count", 0) + 1
        judge_out = state.get("judge_output") or {}
        feedback = judge_out.get("feedback") or ""
        candidate = state.get("candidate_answer", "")
        query = state.get("sanitized_query") or state.get("query", "")
        intent = state.get("intent")
        history = state.get("chat_history")

        regen_dispatch = {
            IntentType.RAG_ONLY: self._regenerate_rag,
            IntentType.HYBRID: self._regenerate_hybrid,
        }

        handler = regen_dispatch.get(intent)
        if handler:
            try:
                new_answer = handler(query, candidate, feedback, history, state)
                if new_answer:
                    return {
                        "candidate_answer": new_answer,
                        "retry_count": retries,
                    }
            except Exception as exc:
                logger.warning("Regeneration error for intent %s: %s", intent, exc)

        return {
            "candidate_answer": candidate,
            "retry_count": retries,
        }


    def output_guardrail_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 7: Output sanitization, PII masking, and citation verification."""
        candidate = state.get("candidate_answer", "")
        intent = state.get("intent")
        citations = state.get("citations")

        guard_res: OutputGuardrailResult = self.output_guardrail.check(
            candidate,
            intent=intent,
            citations=citations,
        )

        return {
            "final_answer": guard_res.sanitized_answer,
            "output_guardrail_result": guard_res,
        }

    def _route_after_input_guardrail(self, state: AgentState) -> str:
        if state.get("is_input_safe") is False:
            return "output_guardrail"
        return "intent_router"

    def _route_after_intent(self, state: AgentState) -> Union[str, List[str]]:
        """
        Dispatches intent using INTENT_ROUTES mapping.
        For independent HYBRID queries, fans out concurrently to ['sql_agent', 'rag_agent'].
        """
        if state.get("is_input_safe") is False:
            return "output_guardrail"

        intent = state.get("intent", IntentType.UNKNOWN)
        return INTENT_ROUTES.get(intent, "answer_synthesis")

    def _route_after_judge(self, state: AgentState) -> str:
        judge_out = state.get("judge_output") or {}
        passed = judge_out.get("passed", True)
        retries = state.get("retry_count", 0)
        max_retries = getattr(self.settings, "MAX_REGENERATION_ATTEMPTS", 2)

        if not passed and retries < max_retries:
            return "regeneration"
        return "output_guardrail"

    def _build_graph(self) -> StateGraph:
        """
        Constructs master LangGraph with parallel fan-out/fan-in for HYBRID queries:
        START -> input_guardrail -> intent_router -> [sql_agent, rag_agent] -> answer_synthesis -> llm_judge -> output_guardrail -> END
        """
        builder = StateGraph(AgentState)

        builder.add_node("input_guardrail", self.input_guardrail_node)
        builder.add_node("intent_router", self.intent_router_node)
        builder.add_node("sql_agent", self.sql_agent_node)
        builder.add_node("rag_agent", self.rag_agent_node)
        builder.add_node("answer_synthesis", self.answer_synthesis_node)
        builder.add_node("llm_judge", self.llm_judge_node)
        builder.add_node("regeneration", self.regeneration_node)
        builder.add_node("output_guardrail", self.output_guardrail_node)

        builder.add_edge(START, "input_guardrail")
        builder.add_conditional_edges(
            "input_guardrail",
            self._route_after_input_guardrail,
            {
                "output_guardrail": "output_guardrail",
                "intent_router": "intent_router",
            },
        )
        builder.add_conditional_edges(
            "intent_router",
            self._route_after_intent,
            {
                "sql_agent": "sql_agent",
                "rag_agent": "rag_agent",
                "answer_synthesis": "answer_synthesis",
                "output_guardrail": "output_guardrail",
            },
        )

        # Parallel fan-in: both branches converge on answer_synthesis
        builder.add_edge("sql_agent", "answer_synthesis")
        builder.add_edge("rag_agent", "answer_synthesis")

        builder.add_edge("answer_synthesis", "llm_judge")
        builder.add_conditional_edges(
            "llm_judge",
            self._route_after_judge,
            {
                "regeneration": "regeneration",
                "output_guardrail": "output_guardrail",
            },
        )
        builder.add_edge("regeneration", "llm_judge")
        builder.add_edge("output_guardrail", END)

        return builder

    @traceable(name="HR-Copilot-Request", run_type="chain")
    def process_query(
        self,
        query: str,
        history: Optional[Union[List[str], List[dict], List[Any]]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        start_time = time.perf_counter()
        active_req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        active_session_id = session_id or "default_session"
        start_ts = datetime.now(timezone.utc).isoformat()

        update_trace_request(
            query=query,
            session_id=active_session_id,
            request_id=active_req_id,
            timestamp=start_ts,
            environment=self.settings.APP_ENV,
            provider=self.settings.DEFAULT_PROVIDER,
            model=self.settings.DEFAULT_MODEL,
        )

        try:
            initial_state: AgentState = {
                "messages": [],
                "query": query,
                "chat_history": history if isinstance(history, list) else None,
                "retry_count": 0,
                "errors": [],
                "metadata": {
                    "session_id": active_session_id,
                    "request_id": active_req_id,
                },
            }

            final_state: AgentState = self.workflow.invoke(initial_state)

            decision: Optional[RouteDecision] = final_state.get("route_decision")
            if decision is None:
                category, target = INTENT_FALLBACK_ROUTES.get(
                    final_state.get("intent"), (RouteCategory.GENERAL, None)
                )
                decision = RouteDecision(
                    category=category,
                    allowed=final_state.get("is_input_safe", True),
                    target=target,
                    reason=final_state.get("intent_reasoning", ""),
                    confidence=1.0,
                )

            source = final_state.get("source") or "master"
            response_text = final_state.get("final_answer") or final_state.get("candidate_answer", "")
            allowed = final_state.get("is_input_safe", True)
            rag_result = final_state.get("rag_result")
            sql_result = final_state.get("sql_result")
            judge_decision = final_state.get("judge_decision")
            error = final_state.get("errors")[-1] if final_state.get("errors") else None

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            status = "blocked" if not allowed else ("error" if error else "success")

            record_root_trace(
                decision=decision,
                response_text=response_text,
                status=status,
                latency_ms=elapsed_ms,
                judge_decision=judge_decision,
                error=error,
            )

            return OrchestratorResponse(
                decision=decision,
                source=source,
                response=response_text,
                allowed=allowed,
                rag_result=rag_result,
                sql_result=sql_result,
                judge_decision=judge_decision,
                error=error,
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = str(exc)
            logger.exception("Unexpected error in process_query: %s", err_msg)
            fallback_err_resp = "An unexpected system error occurred while processing your request. Please try again later."
            fallback_decision = RouteDecision(
                category=RouteCategory.GENERAL,
                allowed=False,
                target=None,
                reason=err_msg,
                confidence=0.0,
            )
            record_root_trace(
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


def create_copilot_graph(
    router: Optional[QueryRouter] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    sql_pipeline: Optional[SQLPipeline] = None,
    input_guardrail: Optional[InputGuardrail] = None,
    output_guardrail: Optional[OutputGuardrail] = None,
    judge: Optional[LLMJudge] = None,
    settings: Optional[Settings] = None,
) -> StateGraph:
    """Factory creating uncompiled StateGraph for the Master Orchestrator."""
    orchestrator = MasterOrchestrator(
        router=router,
        rag_pipeline=rag_pipeline,
        sql_pipeline=sql_pipeline,
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
        judge=judge,
        settings=settings,
    )
    return orchestrator.graph


def get_copilot_graph(
    orchestrator: Optional[MasterOrchestrator] = None,
    settings: Optional[Settings] = None,
) -> Any:
    """Factory creating compiled executable LangGraph workflow for the Master Orchestrator."""
    orch = orchestrator or MasterOrchestrator(settings=settings)
    return orch.workflow
