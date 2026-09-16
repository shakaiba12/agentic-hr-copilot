"""
Dense Vector Retrieval Pipeline for RAG.
Orchestrates Query Embedding -> Chroma Vector Search -> Structured Result Packing.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.config import Settings, get_settings
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder
from src.rag.retrieval.retriever import DenseRetriever, RetrievalResult


class RetrievalPipeline:
    """
    Retrieval Pipeline for Phase 2 Dense Vector Search.
    Encapsulates DenseRetriever with configuration from settings.
    """

    def __init__(
        self,
        retriever: Optional[DenseRetriever] = None,
        embedder: Optional[SharedEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        collection_name: str = "hr_documents",
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=collection_name,
            settings=self.settings,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Execute dense semantic search for the user query.

        Args:
            query: User natural language inquiry.
            top_k: Number of chunks to retrieve (defaults to settings.TOP_K_RETRIEVAL or 5).

        Returns:
            List of RetrievalResult items sorted by nearest distance.
        """
        k = top_k if top_k is not None else getattr(self.settings, "TOP_K_RETRIEVAL", 5)
        return self.retriever.retrieve(query=query, top_k=k)
