"""
Comprehensive Unit Tests for RAG Embedding & ChromaDB Vector Storage Layer.
Verifies Document Embedding, Query Embedding, Shared Model Invariants, and Isolated ChromaDB Storage.
"""

from __future__ import annotations

import pytest
import chromadb
from pathlib import Path
from typing import List

from src.core.config import Settings
from src.rag.chunking import ChunkingConfig, DocumentChunk, StructureAwareChunker
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder
from src.rag.ingestion.pipeline import IngestionPipeline, IngestionSummary


@pytest.fixture(scope="module")
def shared_embedder() -> SharedEmbedder:
    """Fixture providing a single initialized SharedEmbedder instance across embedding tests."""
    return SharedEmbedder()


@pytest.fixture
def isolated_chroma_store(tmp_path: Path) -> ChromaVectorStore:
    """Fixture providing an isolated ChromaVectorStore using an ephemeral test directory."""
    return ChromaVectorStore(persist_dir=tmp_path / "test_chroma_db")


class TestDocumentEmbedding:
    """Unit tests for document chunk embedding."""

    def test_embed_plain_text_documents(self, shared_embedder: SharedEmbedder):
        texts = [
            "Employees are entitled to 20 days of paid annual leave per calendar year.",
            "Health insurance coverage begins on the first day of full-time employment.",
            "The home office setup stipend provides up to $500 for ergonomic equipment.",
        ]
        embeddings = shared_embedder.embed_documents(texts)

        # Vector count equals input count
        assert len(embeddings) == len(texts)

        # Vectors are non-empty float lists
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 384
            assert all(isinstance(val, float) for val in emb)

    def test_embed_document_chunk_objects(self, shared_embedder: SharedEmbedder):
        chunk1 = DocumentChunk(
            chunk_id="test_doc_0001_abcd1234",
            document_id="test_doc",
            text="Standard working hours are 9:00 AM to 5:00 PM Monday through Friday.",
            chunk_index=1,
            heading_path=["Working Hours", "Standard Schedule"],
            metadata={"char_count": 68, "estimated_tokens": 17},
        )
        chunk2 = DocumentChunk(
            chunk_id="test_doc_0002_efgh5678",
            document_id="test_doc",
            text="Overtime must be pre-approved by the department manager in writing.",
            chunk_index=2,
            heading_path=["Working Hours", "Overtime Policy"],
            metadata={"char_count": 68, "estimated_tokens": 17},
        )

        embeddings = shared_embedder.embed_documents([chunk1, chunk2])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_embed_empty_document_list(self, shared_embedder: SharedEmbedder):
        embeddings = shared_embedder.embed_documents([])
        assert embeddings == []

    def test_embed_whitespace_only_documents_raises(self, shared_embedder: SharedEmbedder):
        with pytest.raises(ValueError, match="Cannot embed empty or whitespace-only"):
            shared_embedder.embed_documents(["   ", "\t\n\r", "  "])


class TestQueryEmbedding:
    """Unit tests for user query embedding."""

    def test_embed_valid_query(self, shared_embedder: SharedEmbedder):
        query = "How many annual leaves do employees receive?"
        emb = shared_embedder.embed_query(query)

        assert isinstance(emb, list)
        assert len(emb) == 384
        assert all(isinstance(v, float) for v in emb)

    def test_query_dimension_matches_document_dimension(self, shared_embedder: SharedEmbedder):
        query = "What is the parental leave duration?"
        doc_text = "Parental leave is granted up to 16 consecutive weeks with full salary."

        q_emb = shared_embedder.embed_query(query)
        d_emb = shared_embedder.embed_documents([doc_text])[0]

        assert len(q_emb) == len(d_emb)
        assert len(q_emb) == 384

    def test_embed_empty_query_raises(self, shared_embedder: SharedEmbedder):
        with pytest.raises(ValueError, match="Cannot generate embedding for empty"):
            shared_embedder.embed_query("")

    def test_embed_whitespace_query_raises(self, shared_embedder: SharedEmbedder):
        with pytest.raises(ValueError, match="Cannot generate embedding for empty"):
            shared_embedder.embed_query("    \t\n  ")


class TestSharedModelInvariants:
    """Verifies single model instance and shared configuration across queries and documents."""

    def test_same_underlying_model_instance(self, shared_embedder: SharedEmbedder):
        model1 = shared_embedder.model
        model2 = shared_embedder.model
        assert model1 is model2

    def test_embedder_dimension_property(self, shared_embedder: SharedEmbedder):
        assert shared_embedder.dimension == 384
        assert "all-MiniLM-L6-v2" in shared_embedder.model_name


class TestChromaVectorStore:
    """Unit tests for ChromaDB storage, metadata preservation, and isolated inspection."""

    def test_store_and_retrieve_chunks(
        self,
        shared_embedder: SharedEmbedder,
        isolated_chroma_store: ChromaVectorStore,
    ):
        chunks = [
            DocumentChunk(
                chunk_id="handbook_0001_abc123",
                document_id="handbook",
                text="Employees receive 20 days of annual leave.",
                chunk_index=1,
                heading_path=["Leave Policy", "Annual Leave"],
                content_type="paragraph",
                start_offset=0,
                end_offset=42,
                metadata={
                    "section": "Annual Leave",
                    "parent_sections": ["Leave Policy"],
                    "char_count": 42,
                    "estimated_tokens": 10,
                    "was_split": False,
                },
            ),
            DocumentChunk(
                chunk_id="handbook_0002_def456",
                document_id="handbook",
                text="Sick leave requires a medical certificate after 3 consecutive days.",
                chunk_index=2,
                heading_path=["Leave Policy", "Sick Leave"],
                content_type="paragraph",
                start_offset=44,
                end_offset=112,
                metadata={
                    "section": "Sick Leave",
                    "parent_sections": ["Leave Policy"],
                    "char_count": 68,
                    "estimated_tokens": 17,
                    "was_split": False,
                },
            ),
        ]

        embeddings = shared_embedder.embed_documents(chunks)
        stored_count = isolated_chroma_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection_name="test_collection",
        )

        assert stored_count == 2
        assert isolated_chroma_store.count("test_collection") == 2

        # Verify inspection output
        inspection = isolated_chroma_store.inspect_collection(
            collection_name="test_collection",
            limit=5,
            show_output=False,
        )

        assert inspection["collection_name"] == "test_collection"
        assert inspection["total_records"] == 2
        assert len(inspection["sample_records"]) == 2

        rec1 = next(r for r in inspection["sample_records"] if r["id"] == "handbook_0001_abc123")
        assert rec1["text"] == "Employees receive 20 days of annual leave."
        assert rec1["dimension"] == 384
        assert rec1["metadata"]["document_id"] == "handbook"
        assert rec1["metadata"]["heading_path"] == "Leave Policy > Annual Leave"
        assert rec1["metadata"]["section"] == "Annual Leave"
        assert rec1["metadata"]["was_split"] is False

    def test_store_chunks_mismatched_lengths_raises(
        self,
        isolated_chroma_store: ChromaVectorStore,
    ):
        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                document_id="doc",
                text="Sample text",
                chunk_index=1,
            )
        ]
        with pytest.raises(ValueError, match="Mismatch between number of chunks"):
            isolated_chroma_store.store_chunks(
                chunks=chunks,
                embeddings=[[0.1] * 384, [0.2] * 384],
                collection_name="test_collection",
            )

    def test_store_empty_chunks_returns_zero(
        self,
        isolated_chroma_store: ChromaVectorStore,
    ):
        count = isolated_chroma_store.store_chunks(
            chunks=[],
            embeddings=[],
            collection_name="empty_collection",
        )
        assert count == 0


class TestIngestionPipeline:
    """Integration test for end-to-end ingestion pipeline with temporary documents."""

    def test_ingest_directory_end_to_end(
        self,
        tmp_path: Path,
        shared_embedder: SharedEmbedder,
        isolated_chroma_store: ChromaVectorStore,
    ):
        docs_dir = tmp_path / "test_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        doc1 = docs_dir / "parental-leave.md"
        doc1.write_text(
            "# Parental Leave Policy\n\n"
            "## Eligibility\n\n"
            "Full-time employees with at least 12 months service qualify for 16 weeks paid parental leave.\n\n"
            "## Application Process\n\n"
            "Submit requests at least 30 days in advance through the HR portal.",
            encoding="utf-8",
        )

        pipeline = IngestionPipeline(
            chunker=StructureAwareChunker(config=ChunkingConfig()),
            embedder=shared_embedder,
            vector_store=isolated_chroma_store,
        )

        summary: IngestionSummary = pipeline.ingest_directory(
            docs_dir=docs_dir,
            collection_name="test_pipeline_coll",
        )

        assert summary.files_processed == 1
        assert summary.total_chunks >= 1
        assert summary.total_vectors_stored == summary.total_chunks
        assert summary.errors == []
        assert "parental-leave" in summary.document_ids

        # Inspect Chroma collection to verify storage
        inspection = isolated_chroma_store.inspect_collection(
            collection_name="test_pipeline_coll",
            show_output=False,
        )
        assert inspection["total_records"] == summary.total_chunks
        for record in inspection["sample_records"]:
            assert record["dimension"] == 384
            assert len(record["preview"]) > 0
