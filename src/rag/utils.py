"""
RAG Utility Functions for source extraction and formatting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_rag_sources(sources: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract normalized chunk dictionaries and citation IDs from pipeline sources."""
    if not isinstance(sources, list):
        return [], []

    chunks_payload: List[Dict[str, Any]] = []
    citations: List[str] = []

    for s in sources:
        if isinstance(s, dict):
            chunks_payload.append({
                "text": s.get("content") or s.get("snippet") or s.get("text", ""),
                "source": s.get("document_id") or s.get("source", "Unknown"),
                "section": s.get("section") or "General",
                "score": s.get("score"),
            })
            doc_id = s.get("document_id")
            if doc_id:
                citations.append(str(doc_id))
        elif isinstance(s, str):
            chunks_payload.append({
                "text": "",
                "source": s,
                "section": "General",
            })

    return chunks_payload, citations
