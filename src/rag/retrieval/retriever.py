"""
Dense Vector Retriever for Enterprise RAG Pipeline.
Performs semantic similarity search over stored ChromaDB vector embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import Settings, get_settings
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Individual retrieved chunk with vector distance and metadata."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


class DenseRetriever:
    """
    Dense vector retriever using pre-computed ChromaDB vector index and SharedEmbedder.
    Computes query embeddings and performs nearest-neighbor search.
    """

    def __init__(
        self,
        embedder: Optional[SharedEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        collection_name: str = "hr_documents",
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or SharedEmbedder(settings=self.settings)
        self.vector_store = vector_store or ChromaVectorStore(settings=self.settings)
        self.collection_name = collection_name

    @traceable(name="RAG_Dense_Retrieval", run_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Execute dense vector search for a user query.

        Args:
            query: Raw user query string.
            top_k: Number of nearest chunks to return (default: 5).

        Returns:
            List of RetrievalResult objects sorted by ascending distance.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Step 1: Generate 384-dimensional dense query vector
        query_vector = self.embedder.embed_query(query)

        # Step 2: Query ChromaDB collection
        collection = self.vector_store.get_or_create_collection(
            collection_name=self.collection_name,
        )

        total_in_coll = collection.count()
        if total_in_coll == 0:
            logger.warning("Collection '%s' is empty. Returning 0 results.", self.collection_name)
            return []

        n_results = min(top_k, total_in_coll)

        query_response = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Step 3: Unpack and format results
        results: List[RetrievalResult] = []

        ids = query_response.get("ids", [[]])[0]
        documents = query_response.get("documents", [[]])[0]
        metadatas = query_response.get("metadatas", [[]])[0]
        distances = query_response.get("distances", [[]])[0]

        for i in range(len(ids)):
            doc_id = ids[i]
            doc_text = documents[i] if len(documents) > i else ""
            meta = metadatas[i] if len(metadatas) > i and metadatas[i] is not None else {}
            dist = float(distances[i]) if len(distances) > i and distances[i] is not None else 0.0

            results.append(
                RetrievalResult(
                    id=doc_id,
                    text=doc_text,
                    metadata=meta,
                    distance=dist,
                )
            )

        return results
