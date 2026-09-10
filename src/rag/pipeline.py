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
from src.evaluation.judge import LLMJudge
from src.evaluation.schemas import JudgeDecision, RAGJudgeInput
from src.rag.context_builder import ContextBuilder, FormattedContext
from src.rag.generator import RAGGenerationResult, RAGGenerator, UNANSWERABLE_FALLBACK
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
    judge_decision: Optional[JudgeDecision] = None
    retries_attempted: int = 0


class RAGPipeline:
    """
    Enterprise RAG pipeline handler for company docs and HR policy inquiries.
    Integrates 2-stage retrieval (Dense Vector + Cross-Encoder) with grounded LLM generation
    and post-generation LLM Judge verification.
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

    @traceable(name="RAGPipeline", run_type="chain")
    def handle(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGPipelineResult:
        """
        Execute full RAG pipeline: retrieval -> reranking -> context -> generation -> judge verification.
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
                        "document_ids": [c.metadata.get("document_id", "unknown") for c in reranked_chunks],
                        "scores": [c.reranker_score for c in reranked_chunks],
                    })

            # Step 2: Context Building Span
            with safe_trace_span(
                name="RAG_Context_Builder",
                run_type="parser",
                inputs={"chunks_count": len(reranked_chunks), "chunk_ids": [c.id for c in reranked_chunks]},
            ) as ctx_span:
                formatted_ctx: FormattedContext = self.context_builder.build_context(
                    chunks=reranked_chunks,
                )
                if ctx_span:
                    ctx_span.end(outputs={
                        "chunks_included": formatted_ctx.chunks_included,
                        "token_estimate": formatted_ctx.token_estimate,
                        "context_snippet": formatted_ctx.context_text[:300] if formatted_ctx.context_text else "",
                        "sources": formatted_ctx.sources,
                    })

            # Step 3: Grounded LLM Generation Span
            with safe_trace_span(
                name="RAG_LLM_Generation",
                run_type="chain",
                inputs={"query": cleaned_query, "sources_count": len(formatted_ctx.sources)},
            ) as gen_span:
                gen_result = self.generator.generate(
                    query=cleaned_query,
                    formatted_context=formatted_ctx,
                )
                if gen_span:
                    gen_span.end(outputs={
                        "answer": gen_result.answer,
                        "answer_length": len(gen_result.answer),
                        "grounded": gen_result.grounded,
                        "sources": gen_result.sources,
                    })

            # Step 4: LLM-as-a-Judge Verification & Bounded Regeneration
            decision: Optional[JudgeDecision] = None
            retries = 0
            is_unanswerable = (
                UNANSWERABLE_FALLBACK.lower() in gen_result.answer.lower()
                or "do not provide enough information" in gen_result.answer.lower()
            )

            # Format documents for judge
            docs_payload = []
            for s in formatted_ctx.sources:
                if isinstance(s, dict):
                    docs_payload.append({
                        "source": s.get("document_id") or s.get("document") or s.get("source", "Unknown"),
                        "section": s.get("heading_path") or s.get("section", "General"),
                        "content": s.get("snippet") or s.get("content", ""),
                    })
                else:
                    docs_payload.append({
                        "source": getattr(s, "document_id", getattr(s, "document", getattr(s, "source", "Unknown"))),
                        "section": getattr(s, "heading_path", getattr(s, "section", "General")),
                        "content": getattr(s, "snippet", getattr(s, "content", "")),
                    })

            if self.settings.ENABLE_LLM_JUDGE:
                with safe_trace_span(
                    name="RAG_LLM_Judge",
                    run_type="chain",
                    inputs={"query": cleaned_query, "answer": gen_result.answer, "docs_count": len(docs_payload)},
                ) as judge_span:
                    decision = self.judge.evaluate_rag(
                        RAGJudgeInput(
                            question=cleaned_query,
                            answer=str(gen_result.answer),
                            retrieved_docs=docs_payload,
                            history=history,
                        )
                    )

                    # Bounded regeneration loop if judge fails or flags needs_regeneration
                    max_retries = getattr(self.settings, "MAX_JUDGE_RETRIES", 1)
                    while (not decision.passed or decision.needs_regeneration) and retries < max_retries:
                        retries += 1
                        logger.info(
                            f"RAG Judge rejected answer. Initiating retry {retries}/{max_retries}. "
                            f"Reason: {decision.reason}"
                        )
                        critique = (
                            "; ".join(decision.issues)
                            if decision.issues
                            else decision.reason
                        )
                        with safe_trace_span(
                            name=f"RAG_Regeneration_Attempt_{retries}",
                            run_type="chain",
                            inputs={"query": cleaned_query, "previous_answer": gen_result.answer, "critique": critique},
                        ) as retry_span:
                            regen_res = self.generator.regenerate(
                                query=cleaned_query,
                                formatted_context=formatted_ctx,
                                previous_answer=str(gen_result.answer) if hasattr(gen_result, "answer") else str(gen_result),
                                judge_feedback=critique,
                                history=history,
                            )
                            if isinstance(regen_res, RAGGenerationResult) and isinstance(regen_res.answer, str):
                                gen_result = regen_res
                            elif hasattr(regen_res, "answer") and isinstance(regen_res.answer, str):
                                gen_result = regen_res
                            elif isinstance(regen_res, str):
                                gen_result.answer = regen_res

                            decision = self.judge.evaluate_rag(
                                RAGJudgeInput(
                                    question=cleaned_query,
                                    answer=str(gen_result.answer) if hasattr(gen_result, "answer") else str(gen_result),
                                    retrieved_docs=docs_payload,
                                    history=history,
                                )
                            )
                            decision.retries_attempted = retries
                            if retry_span:
                                retry_span.end(outputs={
                                    "regenerated_answer": gen_result.answer,
                                    "judge_passed": decision.passed,
                                    "score": decision.score,
                                })

                    if judge_span:
                        judge_span.end(outputs={
                            "passed": decision.passed,
                            "score": decision.score,
                            "correctness": getattr(decision, "correctness", decision.score),
                            "groundedness": getattr(decision, "groundedness", 1.0),
                            "relevance": getattr(decision, "relevance", 1.0),
                            "hallucination": getattr(decision, "hallucination", False),
                            "verdict_status": decision.verdict_status,
                            "retries": retries,
                        })

            try:
                from langsmith.run_helpers import get_current_run_tree
                run = get_current_run_tree()
                if run:
                    run.inputs = {"query": cleaned_query}
                    run.outputs = {
                        "response": gen_result.answer,
                        "sources": gen_result.sources,
                        "chunks_count": gen_result.chunks_count,
                        "grounded": gen_result.grounded,
                        "judge_passed": decision.passed if decision else None,
                        "judge_score": decision.score if decision else None,
                    }
            except Exception:
                pass

            return RAGPipelineResult(
                success=True,
                query=cleaned_query,
                response=gen_result.answer,
                sources=gen_result.sources,
                chunks_count=gen_result.chunks_count,
                grounded=gen_result.grounded,
                judge_decision=decision,
                retries_attempted=retries,
            )

        except Exception as e:
            logger.error("RAG Pipeline execution failed: %s", e, exc_info=True)
            return RAGPipelineResult(
                success=False,
                query=cleaned_query,
                response=f"I encountered an error retrieving HR policy information: {e}",
                error=str(e),
            )

