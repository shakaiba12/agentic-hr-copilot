"""
Master Orchestrator and Query Gate built with LangGraph.
Dispatches queries across Input Guardrails, QueryRouter, SQL Agent Subgraph, RAG Agent Subgraph,
Answer Synthesis, LLM Judge, Regeneration, and Output Guardrails in a structured StateGraph.
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
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None

from src.core.config import Settings, get_settings
from src.core.llm import get_llm
from src.core.router import QueryRouter, RouteCategory, RouteDecision
from src.core.state import AgentState, IntentType
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import JudgeDecision, RAGJudgeInput, SQLJudgeInput, VerdictStatus
from src.guardrails.input_guardrail import InputGuardrail, InputGuardrailResult
from src.guardrails.output_guardrail import OutputGuardrail, OutputGuardrailResult
from src.rag.graph import get_compiled_rag_graph
from src.rag.pipeline import RAGPipeline, RAGPipelineResult
from src.sql.graph import get_compiled_sql_graph
from src.sql.pipeline import SQLPipeline, SQLPipelineResult

logger = logging.getLogger(__name__)

INTENT_ROUTES: Dict[IntentType, str] = {
    IntentType.SQL_ONLY: "sql_agent",
    IntentType.RAG_ONLY: "rag_agent",
    IntentType.HYBRID: "sql_agent",
    IntentType.CASUAL: "answer_synthesis",
    IntentType.UNKNOWN: "answer_synthesis",
}

CATEGORY_TO_INTENT: Dict[RouteCategory, tuple[IntentType, str]] = {
    RouteCategory.DATA_QUERY: (IntentType.SQL_ONLY, "sql"),
    RouteCategory.RAG_KNOWLEDGE: (IntentType.RAG_ONLY, "rag"),
    RouteCategory.CASUAL: (IntentType.CASUAL, "master"),
    RouteCategory.GENERAL: (IntentType.CASUAL, "master"),
    RouteCategory.OUT_OF_SCOPE: (IntentType.CASUAL, "master"),
}

INTENT_FALLBACK_ROUTES: Dict[Optional[IntentType], tuple[RouteCategory, Optional[str]]] = {
    IntentType.SQL_ONLY: (RouteCategory.DATA_QUERY, "data_specialist"),
    IntentType.RAG_ONLY: (RouteCategory.RAG_KNOWLEDGE, "rag"),
    IntentType.CASUAL: (RouteCategory.CASUAL, None),
}

STATIC_CATEGORY_RESPONSES: Dict[RouteCategory, str] = {
    RouteCategory.CASUAL: (
        "Hello! I am your Enterprise HR Intelligence Assistant. "
        "I can help with company HR policies, benefits, leave, expenses, and employee information. "
        "How can I help you today?"
    ),
    RouteCategory.OUT_OF_SCOPE: (
        "I'm designed to help with company HR policies, employee information, "
        "benefits, leave, expenses, and related workplace questions. "
        "Your query falls outside of approved enterprise documentation."
    ),
}

DEFAULT_FALLBACK_ANSWER = (
    "I could not determine the specific enterprise department for this question. "
    "Please ask about company HR policies, benefits, leave, or employee counts."
)


def _normalize_rag_sources(sources: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract normalized chunk dictionaries and citation IDs from pipeline sources."""
    if not isinstance(sources, list):
        return [], []

    chunks_payload: List[Dict[str, Any]] = []
    citations: List[str] = []

    for s in sources:
        if isinstance(s, dict):
            chunks_payload.append({
                "text": s.get("content") or s.get("snippet") or s.get("text", ""),
                "source": s.get("document_id") or s.get("source", "Unknown"),
                "section": s.get("section") or "General",
                "score": s.get("score"),
            })
            doc_id = s.get("document_id")
            if doc_id:
                citations.append(str(doc_id))
        elif isinstance(s, str):
            chunks_payload.append({
                "text": "",
                "source": s,
                "section": "General",
            })

    return chunks_payload, citations


def _build_trace_tags(decision: RouteDecision) -> List[str]:
    """Construct observability tags based on routing decision and safety status."""
    pipeline_name = decision.target or "master"
    tags = [pipeline_name, decision.category.value.lower()]
    if not decision.allowed:
        tags.extend(["guardrail", "blocked"])
        if decision.category == RouteCategory.PROMPT_INJECTION:
            tags.append("prompt-injection")
        elif decision.category == RouteCategory.DESTRUCTIVE_ACTION:
            tags.append("destructive-action")
    elif decision.category == RouteCategory.RAG_KNOWLEDGE:
        tags.extend(["rag", "llm-judge"])
    elif decision.category == RouteCategory.DATA_QUERY:
        tags.extend(["sql", "llm-judge"])
    return tags


@dataclass
class OrchestratorResponse:
    decision: RouteDecision
    source: str
    response: str
    allowed: bool = True
    rag_result: Optional[RAGPipelineResult] = None
    sql_result: Optional[SQLPipelineResult] = None
    judge_decision: Optional[JudgeDecision] = None
    error: Optional[str] = None


class MasterOrchestrator:
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

        # Connect executable subgraphs directly
        self.sql_workflow = getattr(self.sql_pipeline, "workflow", None) or get_compiled_sql_graph(
            settings=self.settings,
            schema_provider=getattr(self.sql_pipeline, "schema_provider", None),
            generator=getattr(self.sql_pipeline, "generator", None),
            guardrail=getattr(self.sql_pipeline, "guardrail", None),
            executor=getattr(self.sql_pipeline, "executor", None),
        )

        self.rag_workflow = getattr(self.rag_pipeline, "workflow", None) or get_compiled_rag_graph(
            settings=self.settings,
            retriever=getattr(self.rag_pipeline, "retriever", None),
            context_builder=getattr(self.rag_pipeline, "context_builder", None),
            generator=getattr(self.rag_pipeline, "generator", None),
        )

        self.graph = self._build_graph()
        self.workflow = self.graph.compile()

    def input_guardrail_node(self, state: AgentState) -> Dict[str, Any]:
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
        """Executes the SQL Agent Subgraph via pipeline façade."""
        query = state.get("sanitized_query") or state.get("query", "")

        try:
            sql_res: SQLPipelineResult = self.sql_pipeline.handle(query)

            updates: Dict[str, Any] = {
                "sql_result": sql_res,
                "sql_output": {
                    "success": sql_res.success,
                    "generated_sql": sql_res.generated_sql,
                    "rows": sql_res.rows,
                    "row_count": sql_res.row_count,
                    "message": sql_res.message,
                    "error": sql_res.error,
                },
                "generated_sql": sql_res.generated_sql,
                "sql_data": sql_res.rows,
                "sql_row_count": sql_res.row_count,
                "is_sql_valid": sql_res.validation.is_valid if sql_res.validation else sql_res.success,
            }

            if state.get("intent") == IntentType.SQL_ONLY:
                updates["candidate_answer"] = sql_res.message
                updates["source"] = "sql"

            if not sql_res.success and sql_res.error:
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
        """Executes the RAG Agent Subgraph via pipeline façade."""
        query = state.get("sanitized_query") or state.get("query", "")

        try:
            rag_res: RAGPipelineResult = self.rag_pipeline.handle(query)
            chunks_payload, citations = _normalize_rag_sources(rag_res.sources)

            updates: Dict[str, Any] = {
                "rag_result": rag_res,
                "rag_output": {
                    "success": rag_res.success,
                    "response": rag_res.response,
                    "sources": rag_res.sources,
                    "chunks_count": rag_res.chunks_count,
                    "grounded": rag_res.grounded,
                    "error": rag_res.error,
                },
                "retrieved_chunks": chunks_payload,
                "citations": citations,
            }

            if state.get("intent") == IntentType.RAG_ONLY:
                updates["candidate_answer"] = rag_res.response
                updates["source"] = "rag"

            if not rag_res.success and rag_res.error:
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
        intent = state.get("intent")
        query = state.get("sanitized_query") or state.get("query", "")
        decision: Optional[RouteDecision] = state.get("route_decision")

        if state.get("candidate_answer") and intent not in (IntentType.HYBRID, None):
            return {"candidate_answer": state["candidate_answer"]}

        if intent == IntentType.HYBRID:
            sql_out = state.get("sql_output") or {}
            rag_out = state.get("rag_output") or {}
            sql_msg = sql_out.get("message", "")
            rag_msg = rag_out.get("response", "")
            synth = (
                f"{sql_msg}\n\n{rag_msg}"
                if sql_msg and rag_msg
                else (sql_msg or rag_msg or "No data available for hybrid query.")
            )
            return {"candidate_answer": synth, "source": "hybrid"}

        cat = decision.category if decision else RouteCategory.CASUAL

        # Dynamically generate natural conversational responses using LLM
        try:
            llm_client = get_llm(temperature=0.3)
            prompt = (
                f"You are PeopleQuery AI, an Enterprise HR Intelligence Copilot. "
                f"You specialize in company HR policies, employee records, benefits, and workplace guidelines.\n\n"
                f"User Query: \"{query}\"\n\n"
                f"Please provide a warm, helpful, and concise response to the user."
            )
            llm_res = llm_client.invoke(prompt)
            answer_text = (llm_res.content if hasattr(llm_res, "content") else str(llm_res)).strip()
            if answer_text:
                return {"candidate_answer": answer_text, "source": "master"}
        except Exception as e:
            logger.debug("Dynamic answer synthesis fallback skipped: %s", e)

        if cat in STATIC_CATEGORY_RESPONSES:
            return {"candidate_answer": STATIC_CATEGORY_RESPONSES[cat], "source": "master"}

        if cat == RouteCategory.GENERAL:
            resp_text = (
                f"This question was handled as a general inquiry: '{query}'. "
                "I specialize in enterprise HR policies and company data."
            )
            return {"candidate_answer": resp_text, "source": "master"}

        return {"candidate_answer": DEFAULT_FALLBACK_ANSWER, "source": "master"}

    def llm_judge_node(self, state: AgentState) -> Dict[str, Any]:
        """Single Central LLM Judge owned by the master graph."""
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

        try:
            if intent in (IntentType.RAG_ONLY, IntentType.HYBRID):
                decision = self.judge.evaluate_rag(
                    RAGJudgeInput(
                        question=query,
                        answer=candidate,
                        retrieved_docs=state.get("retrieved_chunks", []) or [],
                        history=state.get("chat_history"),
                    )
                )
            elif intent == IntentType.SQL_ONLY:
                decision = self.judge.evaluate_sql(
                    SQLJudgeInput(
                        question=query,
                        sql=state.get("generated_sql", "") or "",
                        sql_result=state.get("sql_data", []) or [],
                        answer=candidate,
                        history=state.get("chat_history"),
                    )
                )
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

    def regeneration_node(self, state: AgentState) -> Dict[str, Any]:
        """Bounded regeneration node re-invoking generator with judge critique."""
        retries = state.get("retry_count", 0) + 1
        judge_out = state.get("judge_output") or {}
        feedback = judge_out.get("feedback") or ""
        candidate = state.get("candidate_answer", "")
        query = state.get("sanitized_query") or state.get("query", "")
        intent = state.get("intent")
        history = state.get("chat_history")

        generator = getattr(self.rag_pipeline, "generator", None)
        context_builder = getattr(self.rag_pipeline, "context_builder", None)

        if (
            intent == IntentType.RAG_ONLY
            and generator
            and hasattr(generator, "regenerate")
            and context_builder
        ):
            try:
                chunks = state.get("retrieved_chunks") or []
                formatted_ctx = context_builder.build_context(chunks)
                if formatted_ctx:
                    regen_res = generator.regenerate(
                        query=query,
                        formatted_context=formatted_ctx,
                        previous_answer=candidate,
                        judge_feedback=feedback,
                        history=history,
                    )
                    new_answer = (
                        regen_res.answer if hasattr(regen_res, "answer") else str(regen_res)
                    )
                    return {
                        "candidate_answer": new_answer,
                        "retry_count": retries,
                    }
            except Exception as exc:
                logger.warning("RAG generator regeneration error: %s", exc)

        refined_answer = (
            f"{candidate} (Corrected based on evaluation: {feedback})"
            if feedback
            else candidate
        )
        return {
            "candidate_answer": refined_answer,
            "retry_count": retries,
        }

    def output_guardrail_node(self, state: AgentState) -> Dict[str, Any]:
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

    def _route_after_intent(self, state: AgentState) -> str:
        if state.get("is_input_safe") is False:
            return "output_guardrail"
        intent = state.get("intent", IntentType.UNKNOWN)
        return INTENT_ROUTES.get(intent, "answer_synthesis")

    def _route_after_sql(self, state: AgentState) -> str:
        if state.get("intent") == IntentType.HYBRID:
            return "rag_agent"
        return "answer_synthesis"

    def _route_after_judge(self, state: AgentState) -> str:
        judge_out = state.get("judge_output") or {}
        passed = judge_out.get("passed", True)
        retries = state.get("retry_count", 0)
        max_retries = getattr(self.settings, "MAX_REGENERATION_ATTEMPTS", 2)

        if not passed and retries < max_retries:
            return "regeneration"
        return "output_guardrail"

    def _build_graph(self) -> StateGraph:
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
        builder.add_conditional_edges(
            "sql_agent",
            self._route_after_sql,
            {
                "rag_agent": "rag_agent",
                "answer_synthesis": "answer_synthesis",
            },
        )
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

            self._update_root_meta(
                metadata={
                    "intent": decision.category.value,
                    "route": decision.target or "master",
                    "route_category": decision.category.value,
                    "route_target": decision.target,
                    "allowed": decision.allowed,
                    "confidence": decision.confidence,
                    "pipeline": decision.target or "master",
                },
                tags=_build_trace_tags(decision),
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
            self._record_root_output(
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


def create_copilot_graph(
    router: Optional[QueryRouter] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    sql_pipeline: Optional[SQLPipeline] = None,
    input_guardrail: Optional[InputGuardrail] = None,
    output_guardrail: Optional[OutputGuardrail] = None,
    judge: Optional[LLMJudge] = None,
    settings: Optional[Settings] = None,
) -> StateGraph:
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
    orch = orchestrator or MasterOrchestrator(settings=settings)
    return orch.workflow
