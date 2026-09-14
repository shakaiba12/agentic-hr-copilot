"""
ChromaDB Storage & Developer Inspection Wrapper for RAG Vector Store.
Preserves chunk IDs, document text, hierarchical metadata, and dense vector embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.core.config import Settings, get_settings
from src.rag.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB vector store manager.
    Handles persistent storage of DocumentChunks and embeddings, metadata flattening, and inspection.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        client: Optional[ClientAPI] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.persist_dir = str(persist_dir or self.settings.CHROMA_PERSIST_DIR)

        if client is not None:
            self._client = client
        else:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)

    @property
    def client(self) -> ClientAPI:
        """Returns the active ChromaDB client."""
        return self._client

    def get_or_create_collection(self, collection_name: str = "hr_documents") -> Collection:
        """Get existing collection or create a new one with cosine similarity distance metric."""
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        collection_name: str = "hr_documents",
    ) -> int:
        """
        Store a batch of DocumentChunk objects with their pre-computed vector embeddings.

        Args:
            chunks: List of DocumentChunk objects from structure-aware chunking.
            embeddings: Corresponding list of float vector embeddings.
            collection_name: Name of target ChromaDB collection.

        Returns:
            Count of chunks successfully stored.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        collection = self.get_or_create_collection(collection_name=collection_name)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._flatten_metadata(chunk))

        # Upsert ensures idempotent re-ingestion
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info("Successfully stored %d chunks in collection '%s'", len(chunks), collection_name)
        return len(chunks)

    def count(self, collection_name: str = "hr_documents") -> int:
        """Return total number of records stored in the collection."""
        try:
            collection = self._client.get_collection(name=collection_name)
            return collection.count()
        except Exception:
            return 0

    def inspect_collection(
        self,
        collection_name: str = "hr_documents",
        limit: int = 3,
        show_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Inspection helper retrieving collection summary and preview of stored records.
        """
        collection = self.get_or_create_collection(collection_name=collection_name)
        total_records = collection.count()

        records_preview: List[Dict[str, Any]] = []
        if total_records > 0:
            sample = collection.peek(limit=limit)
            ids = sample.get("ids", [])
            documents = sample.get("documents", [])
            metadatas = sample.get("metadatas", [])
            embeddings = sample.get("embeddings", [])

            for i in range(len(ids)):
                emb = embeddings[i] if embeddings is not None and len(embeddings) > i else []
                dim = len(emb) if emb is not None else 0
                preview_vals = [round(float(x), 4) for x in emb[:4]] if emb is not None and len(emb) > 0 else []

                records_preview.append({
                    "id": ids[i],
                    "text": documents[i] if len(documents) > i else "",
                    "metadata": metadatas[i] if len(metadatas) > i else {},
                    "dimension": dim,
                    "preview": preview_vals,
                })

        return {
            "collection_name": collection_name,
            "total_records": total_records,
            "sample_records": records_preview,
        }

    @staticmethod
    def _flatten_metadata(chunk: DocumentChunk) -> Dict[str, Union[str, int, float, bool]]:
        """
        Converts DocumentChunk metadata into flat scalar values supported by ChromaDB
        (str, int, float, bool).
        """
        flat: Dict[str, Union[str, int, float, bool]] = {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content_type": chunk.content_type,
            "heading_path": " > ".join(chunk.heading_path) if chunk.heading_path else "",
        }

        if chunk.start_offset is not None:
            flat["start_offset"] = chunk.start_offset
        if chunk.end_offset is not None:
            flat["end_offset"] = chunk.end_offset

        # Process inner metadata dictionary
        for k, v in chunk.metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif isinstance(v, (list, tuple)):
                flat[k] = ", ".join(str(item) for item in v)
            elif isinstance(v, dict):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v)

        return flat
