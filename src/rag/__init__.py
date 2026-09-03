"""
Enterprise RAG Pipeline Components.
Includes Guardrail Gate, Query Router, Markdown Normalization, Structure-Aware Chunking, and Pipeline Handler.
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
