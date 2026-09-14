"""
RAG Subgraph State definitions.
Structured schemas for Retrieval, Reranking, Context Building, and Grounded Generation.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


class RetrievedChunk(TypedDict, total=False):
    """Metadata and text content of a retrieved document passage."""
    text: str
    source: str
    doc_type: Optional[str]
    section: Optional[str]
    page: Optional[int]
    score: Optional[float]


class RAGState(TypedDict, total=False):
    """Execution state for the LangGraph RAG Agent Subgraph."""
    query: str
    sanitized_query: Optional[str]
    chat_history: Optional[List[Dict[str, Any]]]
    candidate_chunks: Optional[List[Any]]
    reranked_chunks: Optional[List[Any]]
    formatted_context: Optional[Any]
    retrieved_chunks: Optional[List[RetrievedChunk]]
    citations: Optional[List[str]]
    candidate_answer: Optional[str]
    rag_output: Optional[Dict[str, Any]]
    rag_result: Optional[Any]
    errors: Annotated[List[str], operator.add]
