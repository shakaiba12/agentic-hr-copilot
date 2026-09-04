"""
Enterprise RAG Pipeline Components.
Includes Guardrail Gate, Query Router, Markdown Normalization, Structure-Aware Chunking, Ingestion, Retrieval, Reranking, Context Builder, and Grounded Generation.
Uses PEP 562 lazy loading to avoid import warnings when running submodules as CLI entrypoints.
"""

from typing import Any

__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "MarkdownFilter",
    "StructureAwareChunker",
    "RouteCategory",
    "RouteDecision",
    "QueryRouter",
    "RAGPipeline",
    "RAGPipelineResult",
    "SharedEmbedder",
    "ChromaVectorStore",
    "IngestionPipeline",
    "DenseRetriever",
    "RetrievalResult",
    "RetrievalPipeline",
    "CrossEncoderReranker",
    "RerankedResult",
    "RerankedRetriever",
    "ContextBuilder",
    "FormattedContext",
    "RAGGenerator",
    "RAGGenerationResult",
]


def __getattr__(name: str) -> Any:
    if name in ("ChunkingConfig", "DocumentChunk", "MarkdownFilter", "StructureAwareChunker"):
        from src.rag.chunking import (
            ChunkingConfig,
            DocumentChunk,
            MarkdownFilter,
            StructureAwareChunker,
        )
        return locals()[name]
    if name in ("QueryRouter", "RouteCategory", "RouteDecision"):
        from src.rag.router import QueryRouter, RouteCategory, RouteDecision
        return locals()[name]
    if name in ("RAGPipeline", "RAGPipelineResult"):
        from src.rag.pipeline import RAGPipeline, RAGPipelineResult
        return locals()[name]
    if name in ("SharedEmbedder", "ChromaVectorStore", "IngestionPipeline"):
        from src.rag.ingestion import ChromaVectorStore, IngestionPipeline, SharedEmbedder
        return locals()[name]
    if name in ("DenseRetriever", "RetrievalResult", "RetrievalPipeline"):
        from src.rag.retrieval import DenseRetriever, RetrievalPipeline, RetrievalResult
        return locals()[name]
    if name in ("CrossEncoderReranker", "RerankedResult", "RerankedRetriever"):
        from src.rag.reranking import CrossEncoderReranker, RerankedResult, RerankedRetriever
        return locals()[name]
    if name in ("ContextBuilder", "FormattedContext"):
        from src.rag.context_builder import ContextBuilder, FormattedContext
        return locals()[name]
    if name in ("RAGGenerator", "RAGGenerationResult"):
        from src.rag.generator import RAGGenerationResult, RAGGenerator
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
