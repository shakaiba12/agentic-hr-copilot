"""
Shared SentenceTransformer Embedding Component for RAG Ingestion & Querying.
Ensures single model instance and identical configuration for document chunks and user queries.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Union

# Suppress HuggingFace Hub unauthenticated warning and progress bars in terminal
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SharedEmbedder:
    """
    Shared Embedder utilizing SentenceTransformers.
    Guarantees document chunks and queries use the exact same underlying model and embedding dimension.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader ensuring single model instance is initialized and reused."""
        if self._model is None:
            logger.info("Initializing SentenceTransformer embedding model: %s", self.model_name)
            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension of the loaded model (e.g., 384)."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        elif hasattr(self.model, "get_sentence_embedding_dimension"):
            dim = self.model.get_sentence_embedding_dimension()
        else:
            dim = 384
        return int(dim) if dim is not None else 384

    def embed_documents(self, texts: List[Union[str, Any]]) -> List[List[float]]:
        """
        Generate dense vector embeddings for a list of document texts or DocumentChunk objects.

        Args:
            texts: List of strings or chunk objects containing a .text attribute.

        Returns:
            List of float lists representing dense vector embeddings.
        """
        if not texts:
            return []

        # Extract text if DocumentChunk or similar objects are passed
        raw_texts: List[str] = []
        for item in texts:
            if hasattr(item, "text"):
                raw_texts.append(str(item.text))
            elif isinstance(item, str):
                raw_texts.append(item)
            else:
                raw_texts.append(str(item))

        # Validate that we don't encode purely empty/whitespace collections
        if not any(t.strip() for t in raw_texts):
            raise ValueError("Cannot embed empty or whitespace-only document list.")

        embeddings = self.model.encode(
            raw_texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate dense vector embedding for a single user query.

        Args:
            query: Raw user query string.

        Returns:
            List of floats representing the query vector.
        """
        if not query or not query.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only query.")

        query_cleaned = query.strip()
        embedding = self.model.encode(
            query_cleaned,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embedding.tolist()
