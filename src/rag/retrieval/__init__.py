"""
RAG Dense Vector Retrieval Subpackage.
Exposes DenseRetriever, RetrievalResult, and RetrievalPipeline.
"""

from src.rag.retrieval.pipeline import RetrievalPipeline
from src.rag.retrieval.retriever import DenseRetriever, RetrievalResult

__all__ = [
    "DenseRetriever",
    "RetrievalResult",
    "RetrievalPipeline",
]
