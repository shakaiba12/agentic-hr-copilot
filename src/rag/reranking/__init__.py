"""
RAG Cross-Encoder Reranking Subpackage.
Exposes CrossEncoderReranker, RerankedResult, and RerankedRetriever.
"""

from typing import Any

__all__ = [
    "CrossEncoderReranker",
    "RerankedResult",
    "RerankedRetriever",
]


def __getattr__(name: str) -> Any:
    if name in ("CrossEncoderReranker", "RerankedResult"):
        from src.rag.reranking.reranker import CrossEncoderReranker, RerankedResult
        return locals()[name]
    if name == "RerankedRetriever":
        from src.rag.reranking.pipeline import RerankedRetriever
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
