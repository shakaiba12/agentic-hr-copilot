"""
Unit Tests for Dense Vector Retrieval Layer (Phase 2).
Verifies Query Embedding, Top-K Filtering, Result Structure, Error Handling, and Isolated ChromaDB Search.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import List

from src.rag.chunking import DocumentChunk
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder
from src.rag.retrieval.pipeline import RetrievalPipeline
from src.rag.retrieval.retriever import DenseRetriever, RetrievalResult


@pytest.fixture(scope="module")
def shared_embedder() -> SharedEmbedder:
    """Fixture providing a single SharedEmbedder instance."""
    return SharedEmbedder()


@pytest.fixture
def populated_test_retriever(tmp_path: Path, shared_embedder: SharedEmbedder) -> DenseRetriever:
    """
    Creates an isolated ChromaVectorStore with test documents,
    returning a DenseRetriever pointing strictly to the temporary test collection.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_chroma_retrieval")
    collection_name = "test_retrieval_collection"

    chunks = [
        DocumentChunk(
            chunk_id="leave_0001_abc1",
            document_id="time-off-policy",
            text="Employees receive 20 days of paid annual leave per calendar year.",
            chunk_index=1,
            heading_path=["Time Off", "Annual Leave"],
            content_type="paragraph",
            metadata={
                "section": "Annual Leave",
                "parent_sections": ["Time Off"],
                "char_count": 65,
                "estimated_tokens": 16,
                "was_split": False,
            },
        ),
        DocumentChunk(
            chunk_id="leave_0002_abc2",
            document_id="time-off-policy",
            text="Annual leave requests must be submitted at least two weeks in advance via HR portal.",
            chunk_index=2,
            heading_path=["Time Off", "Leave Requests"],
            content_type="paragraph",
            metadata={
                "section": "Leave Requests",
                "parent_sections": ["Time Off"],
                "char_count": 85,
                "estimated_tokens": 21,
                "was_split": False,
            },
        ),
        DocumentChunk(
            chunk_id="parental_0001_def1",
            document_id="parental-leave",
            text="Eligible employees receive 16 weeks of paid parental leave upon birth or adoption.",
            chunk_index=1,
            heading_path=["Parental Leave", "Eligibility"],
            content_type="paragraph",
            metadata={
                "section": "Eligibility",
                "parent_sections": ["Parental Leave"],
                "char_count": 82,
                "estimated_tokens": 20,
                "was_split": False,
            },
        ),
        DocumentChunk(
            chunk_id="benefits_0001_ghi1",
            document_id="benefits-overview",
            text="All full-time US employees and their dependents are eligible for health benefits from day one.",
            chunk_index=1,
            heading_path=["Benefits", "Health Coverage"],
            content_type="paragraph",
            metadata={
                "section": "Health Coverage",
                "parent_sections": ["Benefits"],
                "char_count": 94,
                "estimated_tokens": 23,
                "was_split": False,
            },
        ),
    ]

    embeddings = shared_embedder.embed_documents(chunks)
    store.store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    return DenseRetriever(
        embedder=shared_embedder,
        vector_store=store,
        collection_name=collection_name,
    )


class TestDenseRetriever:
    """Unit tests for dense vector retrieval."""

    def test_retrieve_returns_relevant_chunks(self, populated_test_retriever: DenseRetriever):
        query = "How many days of annual leave do employees receive?"
        results = populated_test_retriever.retrieve(query=query, top_k=2)

        assert len(results) == 2
        # Top result should be the annual leave chunk
        top_result = results[0]
        assert "20 days of paid annual leave" in top_result.text
        assert top_result.metadata["document_id"] == "time-off-policy"
        assert top_result.metadata["section"] == "Annual Leave"
        assert isinstance(top_result.distance, float)
        assert top_result.distance >= 0.0

    def test_top_k_bounds(self, populated_test_retriever: DenseRetriever):
        query = "health benefits coverage"

        res_1 = populated_test_retriever.retrieve(query=query, top_k=1)
        assert len(res_1) == 1

        res_3 = populated_test_retriever.retrieve(query=query, top_k=3)
        assert len(res_3) == 3

        # Requesting more than total available chunks caps at total collection count
        res_all = populated_test_retriever.retrieve(query=query, top_k=50)
        assert len(res_all) == 4

    def test_result_structure_and_types(self, populated_test_retriever: DenseRetriever):
        query = "parental leave policy"
        results = populated_test_retriever.retrieve(query=query, top_k=2)

        for res in results:
            assert isinstance(res, RetrievalResult)
            assert isinstance(res.id, str)
            assert len(res.id) > 0
            assert isinstance(res.text, str)
            assert len(res.text) > 0
            assert isinstance(res.metadata, dict)
            assert isinstance(res.distance, float)
            assert "heading_path" in res.metadata

    def test_results_ordered_by_distance_ascending(self, populated_test_retriever: DenseRetriever):
        query = "paid leave and time off"
        results = populated_test_retriever.retrieve(query=query, top_k=4)

        assert len(results) == 4
        # Cosine distance: smaller distance = more similar
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i + 1].distance

    def test_empty_query_raises_value_error(self, populated_test_retriever: DenseRetriever):
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            populated_test_retriever.retrieve(query="")

        with pytest.raises(ValueError, match="Query string cannot be empty"):
            populated_test_retriever.retrieve(query="   \t\n  ")

    def test_invalid_top_k_raises_value_error(self, populated_test_retriever: DenseRetriever):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            populated_test_retriever.retrieve(query="test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            populated_test_retriever.retrieve(query="test", top_k=-5)

    def test_empty_collection_returns_empty_list(
        self,
        tmp_path: Path,
        shared_embedder: SharedEmbedder,
    ):
        store = ChromaVectorStore(persist_dir=tmp_path / "empty_chroma")
        retriever = DenseRetriever(
            embedder=shared_embedder,
            vector_store=store,
            collection_name="non_existent_collection",
        )
        results = retriever.retrieve(query="any query", top_k=5)
        assert results == []


class TestRetrievalPipeline:
    """Unit tests for RetrievalPipeline wrapper."""

    def test_pipeline_search_delegation(self, populated_test_retriever: DenseRetriever):
        pipeline = RetrievalPipeline(retriever=populated_test_retriever)
        results = pipeline.search("annual leave days", top_k=2)

        assert len(results) == 2
        assert "annual leave" in results[0].text.lower()
