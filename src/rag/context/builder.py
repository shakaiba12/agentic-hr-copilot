"""
RAG Context Builder.
Deduplicates, orders, and formats retrieved/reranked chunks into clean, budgeted prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union

from src.rag.reranking.reranker import RerankedResult
from src.rag.retrieval.retriever import RetrievalResult

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@dataclass
class FormattedContext:
    """Formatted prompt context block along with structured source metadata."""

    context_text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    chunks_included: int = 0
    total_chars: int = 0
    token_estimate: int = 0


class ContextBuilder:
    """
    Builds clean, structured context from retrieved or reranked chunks.
    Ensures deduplication, budget constraints, and metadata preservation.
    """

    def __init__(
        self,
        max_chars: int = 6000,
        max_chunks: int = 5,
    ) -> None:
        self.max_chars = max_chars
        self.max_chunks = max_chunks

    @traceable(name="RAG_Context_Builder", run_type="parser")
    def build_context(
        self,
        chunks: Sequence[Union[RetrievalResult, RerankedResult]],
        max_chars: int | None = None,
        max_chunks: int | None = None,
    ) -> FormattedContext:
        """
        Convert candidate chunks into a clean prompt context block with citations.

        Args:
            chunks: Ordered list of RetrievalResult or RerankedResult items.
            max_chars: Optional character limit override (defaults to self.max_chars).
            max_chunks: Optional chunk count limit override (defaults to self.max_chunks).

        Returns:
            FormattedContext containing formatted string and source tracking.
        """
        char_limit = max_chars or self.max_chars
        chunk_limit = max_chunks or self.max_chunks

        if not chunks:
            return FormattedContext(
                context_text="No relevant company policy documents found.",
                sources=[],
                chunks_included=0,
                total_chars=0,
                token_estimate=0,
            )

        seen_ids: set[str] = set()
        formatted_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []
        accumulated_chars = 0

        for idx, chunk in enumerate(chunks, start=1):
            if len(formatted_blocks) >= chunk_limit:
                break

            # Deduplicate chunks
            if isinstance(chunk, dict):
                chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"chunk_{idx}"
                metadata = chunk.get("metadata") or {}
                doc_id = chunk.get("source") or chunk.get("document_id") or metadata.get("document_id", "unknown_document")
                heading_path = chunk.get("section") or metadata.get("heading_path", "")
                section = chunk.get("section") or metadata.get("section", "")
                content_type = metadata.get("content_type", "section")
                text_body = (chunk.get("text") or chunk.get("content") or chunk.get("snippet") or "").strip()
            else:
                chunk_id = getattr(chunk, "id", f"chunk_{idx}")
                metadata = getattr(chunk, "metadata", {}) or {}
                doc_id = metadata.get("document_id", "unknown_document")
                heading_path = metadata.get("heading_path", "")
                section = metadata.get("section", "")
                content_type = metadata.get("content_type", "section")
                text_body = getattr(chunk, "text", "").strip()

            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            heading_display = " > ".join(heading_path) if isinstance(heading_path, list) else str(heading_path or section)

            block = (
                f"--- [SOURCE {len(formatted_blocks) + 1}] ---\n"
                f"Document: {doc_id}\n"
                f"Section: {heading_display}\n"
                f"Content Type: {content_type}\n"
                f"Chunk ID: {chunk_id}\n\n"
                f"{text_body}\n"
            )

            # Check character budget
            if accumulated_chars + len(block) > char_limit and formatted_blocks:
                break

            formatted_blocks.append(block)
            accumulated_chars += len(block)

            sources.append({
                "source_index": len(formatted_blocks),
                "document_id": doc_id,
                "heading_path": heading_display,
                "section": section,
                "chunk_id": chunk.id,
                "content_type": content_type,
            })

        final_context_text = "\n".join(formatted_blocks)
        token_estimate = max(1, len(final_context_text) // 4)

        return FormattedContext(
            context_text=final_context_text,
            sources=sources,
            chunks_included=len(formatted_blocks),
            total_chars=len(final_context_text),
            token_estimate=token_estimate,
        )
