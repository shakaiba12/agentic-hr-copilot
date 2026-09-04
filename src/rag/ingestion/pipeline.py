"""
RAG Ingestion Pipeline.
Orchestrates: Existing Chunking -> Shared Embedding -> ChromaDB Vector Storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import Settings, get_settings
from src.rag.chunking import ChunkingConfig, DocumentChunk, StructureAwareChunker
from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.embedder import SharedEmbedder

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    """Summary metrics of an ingestion run."""

    files_processed: int = 0
    total_chunks: int = 0
    total_vectors_stored: int = 0
    collection_name: str = "hr_documents"
    document_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class IngestionPipeline:
    """
    End-to-End Document Ingestion Pipeline.
    Connects frozen StructureAwareChunker with SharedEmbedder and ChromaVectorStore.
    """

    def __init__(
        self,
        chunker: Optional[StructureAwareChunker] = None,
        embedder: Optional[SharedEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunker = chunker or StructureAwareChunker(config=ChunkingConfig())
        self.embedder = embedder or SharedEmbedder(settings=self.settings)
        self.vector_store = vector_store or ChromaVectorStore(settings=self.settings)

    def ingest_file(
        self,
        file_path: Path,
        collection_name: str = "hr_documents",
    ) -> List[DocumentChunk]:
        """
        Chunk, embed, and store a single Markdown document.

        Args:
            file_path: Absolute or relative Path to markdown document.
            collection_name: Destination Chroma collection name.

        Returns:
            List of generated DocumentChunk objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        document_id = path.stem

        # Step 1: Existing Chunking (Frozen)
        chunks = self.chunker.chunk_document(raw_text, document_id=document_id)
        if not chunks:
            logger.warning("No chunks generated for %s", path.name)
            return []

        # Step 2: Shared Document Embeddings (384-dim)
        embeddings = self.embedder.embed_documents(chunks)

        # Step 3: ChromaDB Storage
        self.vector_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
        )

        return chunks

    def ingest_directory(
        self,
        docs_dir: Optional[Path] = None,
        collection_name: str = "hr_documents",
    ) -> IngestionSummary:
        """
        Ingest all Markdown files from the specified documents directory.

        Args:
            docs_dir: Directory containing Markdown files (defaults to settings.DOCS_DIR).
            collection_name: Target Chroma collection name.

        Returns:
            IngestionSummary containing counts and processed files.
        """
        target_dir = Path(docs_dir or self.settings.DOCS_DIR)
        summary = IngestionSummary(collection_name=collection_name)

        if not target_dir.exists():
            summary.errors.append(f"Directory not found: {target_dir}")
            return summary

        md_files = sorted(list(target_dir.glob("*.md")))
        if not md_files:
            summary.errors.append(f"No markdown (.md) files found in {target_dir}")
            return summary

        all_chunks: List[DocumentChunk] = []

        for md_file in md_files:
            try:
                chunks = self.ingest_file(file_path=md_file, collection_name=collection_name)
                summary.files_processed += 1
                summary.total_chunks += len(chunks)
                summary.document_ids.append(md_file.stem)
                all_chunks.extend(chunks)
            except Exception as e:
                err_msg = f"Failed to ingest {md_file.name}: {e}"
                logger.error(err_msg)
                summary.errors.append(err_msg)

        summary.total_vectors_stored = self.vector_store.count(collection_name=collection_name)
        return summary
