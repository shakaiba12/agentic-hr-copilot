"""
End-to-End Enterprise RAG Pipeline Handler.
Integrates Vector Retrieval -> Cross-Encoder Reranker -> Context Builder -> Grounded LLM Generation.
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
from src.core.observability import safe_trace_span
from src.rag.context_builder import ContextBuilder, FormattedContext
from src.rag.generator import RAGGenerator, UNANSWERABLE_FALLBACK
from src.rag.reranking.pipeline import RerankedRetriever

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


class RAGPipeline:
    """
    Enterprise RAG pipeline handler for company docs and HR policy inquiries.
    Integrates 2-stage retrieval (Dense Vector + Cross-Encoder) with grounded LLM generation.
    """

    def __init__(
        self,
        settings: Optional[Union[Settings, RerankedRetriever]] = None,
        retriever: Optional[RerankedRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[RAGGenerator] = None,
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

    @traceable(name="RAGPipeline", run_type="chain")
    def handle(self, query: str) -> RAGPipelineResult:
        """
        Execute full RAG pipeline: retrieval -> reranking -> context -> generation.
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
            # Step 1: Retrieval + Reranking Span
            with safe_trace_span(
                name="RAG_TwoStage_Retrieval",
                run_type="retriever",
                inputs={"query": cleaned_query},
            ) as ret_span:
                reranked_chunks = self.retriever.retrieve(
                    query=cleaned_query,
                    retrieval_k=10,
                    rerank_k=getattr(self.settings, "TOP_K_RETRIEVAL", 4),
                )
                if ret_span:
                    ret_span.end(outputs={
                        "chunks_found": len(reranked_chunks),
                        "chunk_ids": [c.id for c in reranked_chunks],
                    })

            # Step 2: Context Building Span
            with safe_trace_span(
                name="RAG_Context_Builder",
                run_type="parser",
                inputs={"chunks_count": len(reranked_chunks)},
            ) as ctx_span:
                formatted_ctx: FormattedContext = self.context_builder.build_context(
                    chunks=reranked_chunks,
                )
                if ctx_span:
                    ctx_span.end(outputs={
                        "chunks_included": formatted_ctx.chunks_included,
                        "token_estimate": formatted_ctx.token_estimate,
                    })

            # Step 3: Grounded LLM Generation Span
            with safe_trace_span(
                name="RAG_LLM_Generation",
                run_type="llm",
                inputs={"query": cleaned_query, "sources_count": len(formatted_ctx.sources)},
            ) as gen_span:
                gen_result = self.generator.generate(
                    query=cleaned_query,
                    formatted_context=formatted_ctx,
                )
                if gen_span:
                    gen_span.end(outputs={
                        "answer_length": len(gen_result.answer),
                        "grounded": gen_result.grounded,
                    })

            return RAGPipelineResult(
                success=True,
                query=cleaned_query,
                response=gen_result.answer,
                sources=gen_result.sources,
                chunks_count=gen_result.chunks_count,
                grounded=gen_result.grounded,
            )

        except Exception as e:
            logger.error("RAG Pipeline execution failed: %s", e, exc_info=True)
            return RAGPipelineResult(
                success=False,
                query=cleaned_query,
                response=f"I encountered an error retrieving HR policy information: {e}",
                error=str(e),
            )
