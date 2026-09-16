"""
End-to-End Enterprise RAG Pipeline Handler.
Acts as a thin compatibility façade around the LangGraph RAG Agent Subgraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.core.config import Settings, get_settings
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import JudgeDecision
from src.rag.context.builder import ContextBuilder
from src.rag.generation.generator import RAGGenerator, UNANSWERABLE_FALLBACK
from src.rag.graph import RAGGraphBuilder, create_rag_graph, get_compiled_rag_graph
from src.rag.reranking.pipeline import RerankedRetriever
from src.rag.state import RAGState

logger = logging.getLogger(__name__)


@dataclass
class RAGPipelineResult:
    """Structured result from RAG Pipeline execution."""

    success: bool
    query: str
    response: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    chunks_count: int = 0
    grounded: bool = True
    error: Optional[str] = None
    judge_decision: Optional[JudgeDecision] = None
    retries_attempted: int = 0


class RAGPipeline:
    """
    Enterprise RAG pipeline handler for company docs and HR policy inquiries.
    Acts as a thin compatibility façade around the LangGraph RAG Agent Subgraph.
    """

    def __init__(
        self,
        settings: Optional[Union[Settings, RerankedRetriever]] = None,
        retriever: Optional[RerankedRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[RAGGenerator] = None,
        judge: Optional[LLMJudge] = None,
    ) -> None:
        if isinstance(settings, RerankedRetriever):
            retriever = settings
            actual_settings = None
        else:
            actual_settings = settings

        self.settings = actual_settings or get_settings()
        self.retriever = retriever or RerankedRetriever(settings=self.settings)
        self.context_builder = context_builder or ContextBuilder(
            max_chars=6000,
            max_chunks=getattr(self.settings, "TOP_K_RETRIEVAL", 4),
        )
        self.generator = generator or RAGGenerator(settings=self.settings)
        self.judge = judge or LLMJudge()

        self._builder = RAGGraphBuilder(
            settings=self.settings,
            retriever=self.retriever,
            context_builder=self.context_builder,
            generator=self.generator,
        )
        self.graph = self._builder.build_graph()
        self.workflow = self.graph.compile()

    @traceable(name="RAGPipeline", run_type="chain")
    def handle(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGPipelineResult:
        """
        Execute full RAG pipeline workflow: retrieval -> reranking -> context -> generation.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RAGPipelineResult(
                success=False,
                query=query,
                response=UNANSWERABLE_FALLBACK,
                error="Empty query",
            )

        try:
            initial_state: RAGState = {
                "query": cleaned_query,
                "sanitized_query": cleaned_query,
                "chat_history": history,
            }

            final_state: RAGState = self.workflow.invoke(initial_state)
            result: Optional[RAGPipelineResult] = final_state.get("rag_result")

            if result is None:
                # Fallback reconstruction if rag_result is unset
                rag_out = final_state.get("rag_output") or {}
                success = rag_out.get("success", True)
                response = rag_out.get("response") or final_state.get("candidate_answer", "")
                sources = rag_out.get("sources", [])
                chunks_count = rag_out.get("chunks_count", len(sources))
                grounded = rag_out.get("grounded", True)
                error = rag_out.get("error")

                result = RAGPipelineResult(
                    success=success,
                    query=cleaned_query,
                    response=response,
                    sources=sources,
                    chunks_count=chunks_count,
                    grounded=grounded,
                    error=error,
                )

            self._record_pipeline_output(
                query=cleaned_query,
                response=result.response,
                sources=result.sources,
                chunks_count=result.chunks_count,
                grounded=result.grounded,
            )
            return result

        except Exception as e:
            logger.error("RAG Pipeline execution failed: %s", e, exc_info=True)
            return RAGPipelineResult(
                success=False,
                query=cleaned_query,
                response=f"I encountered an error retrieving HR policy information: {e}",
                error=str(e),
            )

    @staticmethod
    def _record_pipeline_output(
        query: str,
        response: str,
        sources: List[Any],
        chunks_count: int,
        grounded: bool,
    ) -> None:
        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {"query": query}
                run.outputs = {
                    "response": response,
                    "sources": sources,
                    "chunks_count": chunks_count,
                    "grounded": grounded,
                }
        except Exception:
            pass
