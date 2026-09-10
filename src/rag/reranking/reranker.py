"""
Cross-Encoder Reranker for Enterprise RAG (Phase 4).
Scores and reranks retrieved candidate chunks based on full query-passage cross-attention.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder

from src.core.config import Settings, get_settings
from src.rag.retrieval.retriever import RetrievalResult

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
class RerankedResult:
    """Individual chunk result after cross-encoder reranking."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_distance: float = 0.0
    reranker_score: float = 0.0
    final_rank: int = 1


class CrossEncoderReranker:
    """
    Cross-Encoder Reranking Module.
    Takes candidate chunks from dense retrieval and re-scores them using joint cross-attention.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy initializer ensuring single model instance is loaded."""
        if self._model is None:
            logger.info("Initializing CrossEncoder reranker model: %s", self.model_name)
            self._model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    @traceable(name="RAG_Reranking", run_type="chain")
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        top_n: int = 5,
    ) -> List[RerankedResult]:
        """
        Rerank a list of retrieved chunks against the user query.

        Args:
            query: User search query.
            retrieved_chunks: List of candidate RetrievalResult items from vector search.
            top_n: Number of top reranked chunks to return.

        Returns:
            List of RerankedResult sorted by descending reranker score.
        """
        if not retrieved_chunks:
            return []

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty for reranking.")

        if top_n <= 0:
            raise ValueError(f"top_n must be a positive integer, got {top_n}")

        # Construct (query, passage) pairs for joint cross-attention scoring
        pairs = []
        for chunk in retrieved_chunks:
            meta = chunk.metadata or {}
            heading = meta.get("heading_path", "")
            if isinstance(heading, list):
                heading_str = " > ".join(heading)
            else:
                heading_str = str(heading or meta.get("section", ""))

            passage = f"{heading_str}\n{chunk.text}" if heading_str.strip() else chunk.text
            pairs.append((query.strip(), passage))

        scores = self.model.predict(pairs)

        scored_items: List[RerankedResult] = []
        for i, chunk in enumerate(retrieved_chunks):
            score = float(scores[i])
            scored_items.append(
                RerankedResult(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    retrieval_distance=chunk.distance,
                    reranker_score=score,
                )
            )

        # Sort descending by reranker score (higher score = more relevant)
        scored_items.sort(key=lambda item: item.reranker_score, reverse=True)

        # Assign final rank 1, 2, ... and slice to top_n
        top_results = scored_items[:top_n]
        for rank, item in enumerate(top_results, start=1):
            item.final_rank = rank

        return top_results
