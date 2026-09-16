"""
RAG Ingestion Subpackage.
Includes Structure-Aware Chunking, Shared Embedding, Chroma Vector Storage, and Ingestion Pipeline.
"""

from src.rag.ingestion.chroma_store import ChromaVectorStore
from src.rag.ingestion.chunking import (
    ChunkingConfig,
    DocumentChunk,
    MarkdownFilter,
    StructureAwareChunker,
)
from src.rag.ingestion.embedder import SharedEmbedder
from src.rag.ingestion.pipeline import IngestionPipeline, IngestionSummary

__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "MarkdownFilter",
    "StructureAwareChunker",
    "ChromaVectorStore",
    "SharedEmbedder",
    "IngestionPipeline",
    "IngestionSummary",
]
