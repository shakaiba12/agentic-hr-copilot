"""
Unit Tests for ChromaVectorStore.
Tests metadata flattening, chunk ingestion, validation, and collection operations using mocks.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.rag.chunking import DocumentChunk
from src.rag.ingestion.chroma_store import ChromaVectorStore


class TestChromaVectorStoreMetadataFlattening:
    """Unit tests for metadata flattening and type conversion."""

    def test_flatten_metadata_basic_scalars(self):
        chunk = DocumentChunk(
            chunk_id="test_0001_abcd",
            document_id="policy-doc",
            text="Some policy text",
            chunk_index=1,
            heading_path=["Section A", "Sub A1"],
            content_type="paragraph",
            start_offset=10,
            end_offset=30,
            metadata={
                "char_count": 50,
                "is_active": True,
                "score_float": 0.95,
            },
        )

        flat = ChromaVectorStore._flatten_metadata(chunk)

        assert flat["document_id"] == "policy-doc"
        assert flat["chunk_index"] == 1
        assert flat["content_type"] == "paragraph"
        assert flat["heading_path"] == "Section A > Sub A1"
        assert flat["start_offset"] == 10
        assert flat["end_offset"] == 30
        assert flat["char_count"] == 50
        assert flat["is_active"] is True
        assert flat["score_float"] == 0.95

    def test_flatten_metadata_list_and_dict_handling(self):
        chunk = DocumentChunk(
            chunk_id="test_0002_efgh",
            document_id="policy-doc",
            text="Text with complex metadata",
            chunk_index=2,
            heading_path=[],
            content_type="section",
            metadata={
                "tags": ["tag1", "tag2", "tag3"],
                "nested_obj": {"author": "hr_team", "version": 2},
                "none_field": None,
            },
        )

        flat = ChromaVectorStore._flatten_metadata(chunk)

        assert flat["heading_path"] == ""
        assert flat["tags"] == "tag1, tag2, tag3"
        assert json.loads(flat["nested_obj"]) == {"author": "hr_team", "version": 2}
        assert "none_field" not in flat


class TestChromaVectorStoreOperationsWithMocks:
    """Unit tests for ChromaVectorStore CRUD operations with mocked ChromaDB client."""

    @pytest.fixture
    def mock_chroma_client(self):
        client = MagicMock()
        mock_collection = MagicMock()
        client.get_or_create_collection.return_value = mock_collection
        client.get_collection.return_value = mock_collection
        return client, mock_collection

    def test_store_chunks_successful_upsert(self, mock_chroma_client):
        client, mock_collection = mock_chroma_client
        store = ChromaVectorStore(client=client)

        chunks = [
            DocumentChunk(
                chunk_id="c1",
                document_id="doc1",
                text="Text 1",
                chunk_index=1,
            ),
            DocumentChunk(
                chunk_id="c2",
                document_id="doc1",
                text="Text 2",
                chunk_index=2,
            ),
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        count = store.store_chunks(chunks=chunks, embeddings=embeddings, collection_name="test_col")

        assert count == 2
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["c1", "c2"]
        assert call_kwargs["documents"] == ["Text 1", "Text 2"]
        assert call_kwargs["embeddings"] == embeddings
        assert len(call_kwargs["metadatas"]) == 2

    def test_store_chunks_empty_list_returns_zero(self, mock_chroma_client):
        client, mock_collection = mock_chroma_client
        store = ChromaVectorStore(client=client)

        count = store.store_chunks(chunks=[], embeddings=[], collection_name="test_col")
        assert count == 0
        mock_collection.upsert.assert_not_called()

    def test_store_chunks_length_mismatch_raises_error(self, mock_chroma_client):
        client, _ = mock_chroma_client
        store = ChromaVectorStore(client=client)

        chunks = [DocumentChunk(chunk_id="c1", document_id="d1", text="t1", chunk_index=1)]
        embeddings = [[0.1], [0.2]]

        with pytest.raises(ValueError, match="Mismatch between number of chunks"):
            store.store_chunks(chunks=chunks, embeddings=embeddings)

    def test_count_with_mock(self, mock_chroma_client):
        client, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 42

        store = ChromaVectorStore(client=client)
        assert store.count("test_col") == 42
        client.get_collection.assert_called_with(name="test_col")

    def test_count_exception_returns_zero(self, mock_chroma_client):
        client, _ = mock_chroma_client
        client.get_collection.side_effect = Exception("Collection not found")

        store = ChromaVectorStore(client=client)
        assert store.count("nonexistent") == 0

    def test_inspect_collection_structured_output(self, mock_chroma_client):
        client, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 1
        mock_collection.peek.return_value = {
            "ids": ["c1"],
            "documents": ["Text preview"],
            "metadatas": [{"document_id": "doc1"}],
            "embeddings": [[0.12345, 0.67891]],
        }

        store = ChromaVectorStore(client=client)
        info = store.inspect_collection(collection_name="hr_documents", limit=1)

        assert info["collection_name"] == "hr_documents"
        assert info["total_records"] == 1
        assert len(info["sample_records"]) == 1
        assert info["sample_records"][0]["id"] == "c1"
        assert info["sample_records"][0]["dimension"] == 2
