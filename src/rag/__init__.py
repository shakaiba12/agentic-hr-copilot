"""
Enterprise RAG Pipeline Package.
Subpackages: ingestion, retrieval, reranking, context, generation, evaluation.
Top-level exports: graph, pipeline, state.
"""

from typing import Any

__all__ = [
    # Top-level Subgraph and Pipeline
    "RAGPipeline",
    "RAGPipelineResult",
    "RAGGraphBuilder",
    "create_rag_graph",
    "get_compiled_rag_graph",
    "RAGState",
    "RetrievedChunk",
    # Ingestion
    "ChunkingConfig",
    "DocumentChunk",
    "MarkdownFilter",
    "StructureAwareChunker",
    "SharedEmbedder",
    "ChromaVectorStore",
    "IngestionPipeline",
    "IngestionSummary",
    # Retrieval
    "DenseRetriever",
    "RetrievalResult",
    "RetrievalPipeline",
    # Reranking
    "CrossEncoderReranker",
    "RerankedResult",
    "RerankedRetriever",
    # Context
    "ContextBuilder",
    "FormattedContext",
    # Generation
    "RAGGenerator",
    "RAGGenerationResult",
    "UNANSWERABLE_FALLBACK",
    # Routing compatibility
    "QueryRouter",
    "RouteCategory",
    "RouteDecision",
]


def __getattr__(name: str) -> Any:
    if name in ("RAGGraphBuilder", "create_rag_graph", "get_compiled_rag_graph"):
        from src.rag.graph import RAGGraphBuilder, create_rag_graph, get_compiled_rag_graph
        return locals()[name]
    if name in ("RAGPipeline", "RAGPipelineResult"):
        from src.rag.pipeline import RAGPipeline, RAGPipelineResult
        return locals()[name]
    if name in ("RAGState", "RetrievedChunk"):
        from src.rag.state import RAGState, RetrievedChunk
        return locals()[name]
    if name in ("ChunkingConfig", "DocumentChunk", "MarkdownFilter", "StructureAwareChunker"):
        from src.rag.ingestion.chunking import (
            ChunkingConfig,
            DocumentChunk,
            MarkdownFilter,
            StructureAwareChunker,
        )
        return locals()[name]
    if name in ("SharedEmbedder", "ChromaVectorStore", "IngestionPipeline", "IngestionSummary"):
        from src.rag.ingestion import (
            ChromaVectorStore,
            IngestionPipeline,
            IngestionSummary,
            SharedEmbedder,
        )
        return locals()[name]
    if name in ("DenseRetriever", "RetrievalResult", "RetrievalPipeline"):
        from src.rag.retrieval import DenseRetriever, RetrievalPipeline, RetrievalResult
        return locals()[name]
    if name in ("CrossEncoderReranker", "RerankedResult", "RerankedRetriever"):
        from src.rag.reranking import CrossEncoderReranker, RerankedResult, RerankedRetriever
        return locals()[name]
    if name in ("ContextBuilder", "FormattedContext"):
        from src.rag.context.builder import ContextBuilder, FormattedContext
        return locals()[name]
    if name in ("RAGGenerator", "RAGGenerationResult", "UNANSWERABLE_FALLBACK"):
        from src.rag.generation.generator import (
            RAGGenerationResult,
            RAGGenerator,
            UNANSWERABLE_FALLBACK,
        )
        return locals()[name]
    if name in ("QueryRouter", "RouteCategory", "RouteDecision"):
        from src.core.router import QueryRouter, RouteCategory, RouteDecision
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
