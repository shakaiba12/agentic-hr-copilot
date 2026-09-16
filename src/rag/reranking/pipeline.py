"""
Two-Stage Retrieval Pipeline: Dense Vector Search + Cross-Encoder Reranking.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.config import Settings, get_settings
from src.rag.reranking.reranker import CrossEncoderReranker, RerankedResult
from src.rag.retrieval.retriever import DenseRetriever


class RerankedRetriever:
    """
    Two-stage retriever combining fast dense vector search with precise Cross-Encoder reranking.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        collection_name: str = "hr_documents",
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dense_retriever = dense_retriever or DenseRetriever(
            collection_name=collection_name,
            settings=self.settings,
        )
        self.reranker = reranker or CrossEncoderReranker(settings=self.settings)

    def retrieve(
        self,
        query: str,
        retrieval_k: int = 10,
        rerank_k: int = 5,
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        Execute two-stage search:
        1. Fetch top retrieval_k candidate chunks from ChromaDB.
        2. Rerank candidates with CrossEncoder and return top rerank_k results.

        Args:
            query: User search query.
            retrieval_k: Number of candidates fetched from vector search (default: 10).
            rerank_k: Number of final reranked chunks to return (default: 5).
            top_k: Optional alias for rerank_k.

        Returns:
            List of RerankedResult sorted by descending relevance.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        final_k = top_k if top_k is not None else rerank_k
        cand_k = max(retrieval_k, final_k * 2)

        # Stage 1: Dense Vector Retrieval (Top-K candidates)
        candidates = self.dense_retriever.retrieve(query=query, top_k=cand_k)
        if not candidates:
            return []

        # Stage 2: Cross-Encoder Joint Reranking
        return self.reranker.rerank(
            query=query,
            retrieved_chunks=candidates,
            top_n=final_k,
        )
