"""
LangGraph RAG Agent Subgraph.
Orchestrates: Retrieval -> Reranking -> Context Building -> Grounded Answer Generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from langgraph.graph import END, START, StateGraph

from src.core.config import Settings, get_settings
from src.core.observability import safe_trace_span
from src.rag.context.builder import ContextBuilder, FormattedContext
from src.rag.generation.generator import RAGGenerationResult, RAGGenerator, UNANSWERABLE_FALLBACK
from src.rag.reranking.pipeline import RerankedRetriever
from src.rag.reranking.reranker import CrossEncoderReranker, RerankedResult
from src.rag.retrieval.retriever import DenseRetriever, RetrievalResult
from src.rag.state import RAGState, RetrievedChunk

logger = logging.getLogger(__name__)


def _is_mock(obj: Any) -> bool:
    """Check if an object is a test mock to prevent dynamic attribute auto-creation."""
    return hasattr(obj, "_mock_methods") or hasattr(obj, "_mock_name")


def _resolve_dense_retriever(retriever: Any, dense_retriever: Optional[DenseRetriever], settings: Settings) -> Any:
    """Resolve the appropriate dense vector retriever instance."""
    if dense_retriever is not None:
        return dense_retriever
    if isinstance(retriever, DenseRetriever):
        return retriever
    if hasattr(retriever, "dense_retriever") and not _is_mock(retriever):
        return retriever.dense_retriever
    return retriever or DenseRetriever(settings=settings)


def _resolve_reranker(retriever: Any, reranker: Optional[CrossEncoderReranker], settings: Settings) -> Any:
    """Resolve the appropriate cross-encoder reranker instance."""
    if reranker is not None:
        return reranker
    if hasattr(retriever, "reranker") and not _is_mock(retriever):
        return retriever.reranker
    return CrossEncoderReranker(settings=settings)


def _to_retrieved_chunk(chunk: Any) -> RetrievedChunk:
    """Normalize a candidate or reranked chunk into standard RetrievedChunk state schema."""
    if isinstance(chunk, dict):
        return chunk

    meta = getattr(chunk, "metadata", {}) or {}
    heading = meta.get("heading_path")
    return {
        "text": getattr(chunk, "text", ""),
        "source": meta.get("document_id") or meta.get("document") or meta.get("source", "Unknown"),
        "doc_type": meta.get("content_type", "document"),
        "section": " > ".join(heading) if isinstance(heading, list) else meta.get("section", "General"),
        "page": meta.get("page"),
        "score": getattr(chunk, "reranker_score", None),
    }


def _extract_citations(sources: Sequence[Any]) -> List[str]:
    """Extract document ID citations from source metadata list."""
    return [s["document_id"] for s in sources if isinstance(s, dict) and s.get("document_id")]


class RAGGraphBuilder:
    """Builder for the LangGraph RAG Agent Subgraph."""

    def __init__(
        self,
        settings: Optional[Union[Settings, RerankedRetriever]] = None,
        retriever: Optional[Union[RerankedRetriever, DenseRetriever]] = None,
        context_builder: Optional[ContextBuilder] = None,
        generator: Optional[RAGGenerator] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.settings = settings if isinstance(settings, Settings) else get_settings()
        target_retriever = settings if isinstance(settings, RerankedRetriever) else retriever

        self.dense_retriever = _resolve_dense_retriever(target_retriever, dense_retriever, self.settings)
        self.reranker = _resolve_reranker(target_retriever, reranker, self.settings)
        self.retriever = target_retriever or RerankedRetriever(
            dense_retriever=self.dense_retriever if isinstance(self.dense_retriever, DenseRetriever) else None,
            reranker=self.reranker if isinstance(self.reranker, CrossEncoderReranker) else None,
            settings=self.settings,
        )
        self.context_builder = context_builder or ContextBuilder(
            max_chars=6000,
            max_chunks=getattr(self.settings, "TOP_K_RETRIEVAL", 4),
        )
        self.generator = generator or RAGGenerator(settings=self.settings)

    def retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Dense vector search for top candidate chunks."""
        query = (state.get("sanitized_query") or state.get("query", "")).strip()
        if not query:
            return {"candidate_chunks": []}

        cand_k = max(getattr(self.settings, "RETRIEVAL_K", 10), getattr(self.settings, "TOP_K_RETRIEVAL", 4) * 2)
        with safe_trace_span(name="RAG_Dense_Retrieval", run_type="retriever", inputs={"query": query, "top_k": cand_k}) as span:
            chunks = self.dense_retriever.retrieve(query=query, top_k=cand_k) or []
            if span:
                span.end(outputs={"chunks_found": len(chunks), "chunk_ids": [getattr(c, "id", "") for c in chunks]})

        return {"candidate_chunks": chunks}

    def reranking_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 2: CrossEncoder reranking and state formatting into RetrievedChunk."""
        query = (state.get("sanitized_query") or state.get("query", "")).strip()
        chunks: List[Any] = state.get("candidate_chunks") or []
        top_k = getattr(self.settings, "TOP_K_RETRIEVAL", 4)

        if not chunks or not query:
            return {"reranked_chunks": [], "retrieved_chunks": []}

        if all(isinstance(c, RerankedResult) for c in chunks):
            ranked = chunks[:top_k]
        else:
            with safe_trace_span(name="RAG_Reranking", run_type="chain", inputs={"query": query, "top_n": top_k}) as span:
                ranked = self.reranker.rerank(query=query, retrieved_chunks=chunks, top_n=top_k) or []
                if span:
                    span.end(outputs={"chunks_reranked": len(ranked), "chunk_ids": [getattr(r, "id", "") for r in ranked]})

        return {
            "reranked_chunks": ranked,
            "retrieved_chunks": [_to_retrieved_chunk(c) for c in ranked],
        }

    def context_building_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 3: Assemble deduplicated context and citation sources."""
        chunks = state.get("reranked_chunks") or []
        with safe_trace_span(name="RAG_Context_Builder", run_type="parser", inputs={"chunks_count": len(chunks)}) as span:
            ctx: FormattedContext = self.context_builder.build_context(chunks=chunks)
            if span:
                span.end(outputs={"chunks_included": ctx.chunks_included, "token_estimate": ctx.token_estimate, "sources": ctx.sources})

        return {"formatted_context": ctx, "citations": _extract_citations(ctx.sources)}

    def answer_generation_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 4: Grounded LLM generation from formatted context."""
        from src.rag.pipeline import RAGPipelineResult

        query = (state.get("sanitized_query") or state.get("query", "")).strip()
        if not query:
            empty_res = RAGPipelineResult(success=False, query="", response=UNANSWERABLE_FALLBACK, error="Empty query")
            return {
                "candidate_answer": UNANSWERABLE_FALLBACK,
                "rag_result": empty_res,
                "rag_output": {"success": False, "response": UNANSWERABLE_FALLBACK, "error": "Empty query"},
            }

        ctx: FormattedContext = state.get("formatted_context") or self.context_builder.build_context([])
        with safe_trace_span(name="RAG_LLM_Generation", run_type="chain", inputs={"query": query, "sources_count": len(ctx.sources)}) as span:
            gen_result: RAGGenerationResult = self.generator.generate(query=query, formatted_context=ctx)
            if span:
                span.end(outputs={"answer": gen_result.answer, "grounded": gen_result.grounded, "sources": gen_result.sources})

        citations = _extract_citations(gen_result.sources)
        pipeline_res = RAGPipelineResult(
            success=True,
            query=query,
            response=gen_result.answer,
            sources=gen_result.sources,
            chunks_count=gen_result.chunks_count,
            grounded=gen_result.grounded,
        )

        return {
            "candidate_answer": gen_result.answer,
            "citations": citations,
            "retrieved_chunks": state.get("retrieved_chunks") or [],
            "rag_result": pipeline_res,
            "rag_output": {
                "success": True,
                "response": gen_result.answer,
                "sources": gen_result.sources,
                "chunks_count": gen_result.chunks_count,
                "grounded": gen_result.grounded,
                "error": None,
            },
        }

    def build_graph(self) -> StateGraph:
        """Construct the LangGraph RAG Agent Subgraph."""
        builder = StateGraph(RAGState)
        nodes = ["rag_retrieval", "rag_reranking", "rag_context_building", "rag_answer_generation"]
        funcs = [self.retrieval_node, self.reranking_node, self.context_building_node, self.answer_generation_node]

        for name, fn in zip(nodes, funcs):
            builder.add_node(name, fn)

        builder.add_edge(START, nodes[0])
        for src, dst in zip(nodes[:-1], nodes[1:]):
            builder.add_edge(src, dst)
        builder.add_edge(nodes[-1], END)

        return builder


def create_rag_graph(
    settings: Optional[Union[Settings, RerankedRetriever]] = None,
    retriever: Optional[Union[RerankedRetriever, DenseRetriever]] = None,
    context_builder: Optional[ContextBuilder] = None,
    generator: Optional[RAGGenerator] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> StateGraph:
    """Factory creating uncompiled StateGraph for the RAG Agent."""
    return RAGGraphBuilder(
        settings=settings,
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        dense_retriever=dense_retriever,
        reranker=reranker,
    ).build_graph()


def get_compiled_rag_graph(
    settings: Optional[Union[Settings, RerankedRetriever]] = None,
    retriever: Optional[Union[RerankedRetriever, DenseRetriever]] = None,
    context_builder: Optional[ContextBuilder] = None,
    generator: Optional[RAGGenerator] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> Any:
    """Factory creating compiled executable LangGraph workflow for the RAG Agent."""
    return create_rag_graph(
        settings=settings,
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        dense_retriever=dense_retriever,
        reranker=reranker,
    ).compile()
