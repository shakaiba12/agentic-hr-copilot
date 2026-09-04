"""
Unit Tests for Cross-Encoder Reranker & Two-Stage Retrieval (Phase 4).
Verifies Joint Scoring, Sorting Invariants, Top-N Slicing, and Candidate Preservation.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import List

from src.rag.chunking import DocumentChunk
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder
from src.rag.reranking.pipeline import RerankedRetriever
from src.rag.reranking.reranker import CrossEncoderReranker, RerankedResult
from src.rag.retrieval.retriever import DenseRetriever, RetrievalResult


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


class TestCrossEncoderReranker:
    """Unit tests for CrossEncoderReranker."""

    def test_rerank_orders_by_relevance_descending(self, reranker: CrossEncoderReranker):
        query = "parental leave policy duration"
        candidates = [
            RetrievalResult(
                id="doc_irrelevant",
                text="The company holiday schedule includes Memorial Day and Labor Day.",
                metadata={"section": "Holidays"},
                distance=0.65,
            ),
            RetrievalResult(
                id="doc_relevant",
                text="Eligible employees receive 16 weeks of paid parental leave upon birth or adoption.",
                metadata={"section": "Parental Leave"},
                distance=0.45,
            ),
            RetrievalResult(
                id="doc_semi_relevant",
                text="Employees on leave should notify their manager in advance.",
                metadata={"section": "General Leave"},
                distance=0.55,
            ),
        ]

        reranked = reranker.rerank(query=query, retrieved_chunks=candidates, top_n=3)

        assert len(reranked) == 3
        # Top result must be the parental leave chunk
        assert reranked[0].id == "doc_relevant"
        assert reranked[0].final_rank == 1
        assert reranked[0].reranker_score > reranked[1].reranker_score
        assert reranked[1].reranker_score > reranked[2].reranker_score

        # Ensure retrieval distance is preserved
        assert reranked[0].retrieval_distance == 0.45

    def test_rerank_top_n_bounds(self, reranker: CrossEncoderReranker):
        query = "health benefits"
        candidates = [
            RetrievalResult(id=f"doc_{i}", text=f"Health benefit policy snippet {i}", metadata={}, distance=0.1 * i)
            for i in range(10)
        ]

        top_3 = reranker.rerank(query=query, retrieved_chunks=candidates, top_n=3)
        assert len(top_3) == 3
        for idx, res in enumerate(top_3, start=1):
            assert res.final_rank == idx

    def test_rerank_empty_candidates_returns_empty(self, reranker: CrossEncoderReranker):
        assert reranker.rerank(query="anything", retrieved_chunks=[]) == []

    def test_rerank_empty_query_raises(self, reranker: CrossEncoderReranker):
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            reranker.rerank(
                query="",
                retrieved_chunks=[RetrievalResult(id="1", text="some text", metadata={}, distance=0.1)],
            )

    def test_rerank_invalid_top_n_raises(self, reranker: CrossEncoderReranker):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            reranker.rerank(
                query="valid query",
                retrieved_chunks=[RetrievalResult(id="1", text="some text", metadata={}, distance=0.1)],
                top_n=0,
            )

    def test_rerank_with_mocked_model(self):
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.5]

        custom_reranker = CrossEncoderReranker()
        custom_reranker._model = mock_model

        candidates = [
            RetrievalResult(id="c1", text="text 1", metadata={"heading_path": ["H1", "H2"]}, distance=0.3),
            RetrievalResult(id="c2", text="text 2", metadata={"section": "Sec 2"}, distance=0.2),
            RetrievalResult(id="c3", text="text 3", metadata={}, distance=0.1),
        ]

        results = custom_reranker.rerank(query="test query", retrieved_chunks=candidates, top_n=2)

        assert len(results) == 2
        assert results[0].id == "c2"
        assert results[0].final_rank == 1
        assert results[0].reranker_score == 0.9
        assert results[1].id == "c3"
        assert results[1].final_rank == 2
        assert results[1].reranker_score == 0.5

    def test_rerank_single_candidate(self):
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.85]

        custom_reranker = CrossEncoderReranker()
        custom_reranker._model = mock_model

        candidates = [RetrievalResult(id="single", text="solo text", metadata={}, distance=0.1)]
        results = custom_reranker.rerank(query="query", retrieved_chunks=candidates, top_n=5)

        assert len(results) == 1
        assert results[0].id == "single"
        assert results[0].final_rank == 1
        assert results[0].reranker_score == 0.85


class TestRerankedRetrieverPipeline:
    """Integration test for two-stage retriever in isolated Chroma test collection."""

    def test_two_stage_retrieval(
        self,
        tmp_path: Path,
        reranker: CrossEncoderReranker,
    ):
        embedder = SharedEmbedder()
        store = ChromaVectorStore(persist_dir=tmp_path / "test_chroma_rerank")
        collection_name = "test_rerank_coll"

        chunks = [
            DocumentChunk(
                chunk_id="chunk_pto_1",
                document_id="time-off",
                text="Employees are encouraged to take a minimum of 30 days total paid time off for rest per year.",
                chunk_index=1,
                heading_path=["Time Off", "Paid Time Off"],
                metadata={"section": "Paid Time Off"},
            ),
            DocumentChunk(
                chunk_id="chunk_conduct_1",
                document_id="conduct",
                text="Respect and honesty are expected in all workplace communications.",
                chunk_index=1,
                heading_path=["Conduct"],
                metadata={"section": "Standards"},
            ),
            DocumentChunk(
                chunk_id="chunk_pto_2",
                document_id="time-off",
                text="Please record all of your PTO in Deel at least 5 business days in advance.",
                chunk_index=2,
                heading_path=["Time Off", "Submitting PTO"],
                metadata={"section": "Submitting PTO"},
            ),
        ]

        embeddings = embedder.embed_documents(chunks)
        store.store_chunks(chunks=chunks, embeddings=embeddings, collection_name=collection_name)

        dense_retriever = DenseRetriever(
            embedder=embedder,
            vector_store=store,
            collection_name=collection_name,
        )

        two_stage = RerankedRetriever(
            dense_retriever=dense_retriever,
            reranker=reranker,
            collection_name=collection_name,
        )

        results = two_stage.retrieve(
            query="How much PTO or paid vacation should I take for rest?",
            retrieval_k=3,
            rerank_k=2,
        )

        assert len(results) == 2
        # Top result should be the 30 days PTO chunk
        assert results[0].id == "chunk_pto_1"
        assert results[0].final_rank == 1
        assert "minimum of 30 days" in results[0].text
