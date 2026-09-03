"""
Shared state definitions for PeopleQuery AI Agentic HR Intelligence Copilot.
Streamlined, modular state schema for LangGraph workflows and orchestrators.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

try:
    from langgraph.graph.message import add_messages
except ImportError:
    add_messages = operator.add


class IntentType(str, Enum):
    """Classification intent types for query routing."""
    SQL_ONLY = "SQL_ONLY"
    RAG_ONLY = "RAG_ONLY"
    HYBRID = "HYBRID"
    CASUAL = "CASUAL"
    UNKNOWN = "UNKNOWN"


class RetrievedChunk(TypedDict, total=False):
    """Metadata and text content of a retrieved document passage."""
    text: str
    source: str
    doc_type: str
    section: Optional[str]
    page: Optional[int]
    score: Optional[float]


class JudgeEvaluation(TypedDict, total=False):
    """Structured output from the LLM Judge / Evaluator."""
    decision: str  # "PASS" | "FAIL"
    score: float   # 0.0 - 1.0 overall
    correctness: float
    relevance: float
    faithfulness: float
    completeness: float
    safety: float
    issues: List[str]
    feedback: Optional[str]


class AgentState(TypedDict, total=False):
    """
    Optimized, compact shared state for the Agentic HR Copilot workflow.

    Core Conversational State:
    - messages: LangGraph message history with auto-reducer (add_messages).
    - query: Current raw or sanitized user question.
    - intent: Detected routing category / intent type.
    - next_step: Next agent or node to execute.

    Modular Sub-Pipeline Outputs:
    - routing_output: Output from the router (category, allowed, target, reason, confidence).
    - sql_output: Structured SQL pipeline output (generated_sql, rows, row_count, validation).
    - rag_output: Structured RAG pipeline output (retrieved_chunks, citations, sources).
    - judge_output: Evaluation score, pass/fail status, and self-correction feedback.

    Final Delivery & Session Memory:
    - final_answer: Verified response delivered to the user.
    - chat_history: Past conversation turns for multi-turn contextual memory.
    - retry_count: Self-correction evaluation retry attempts.
    - errors: Accumulated execution errors.
    - metadata: Execution latency, model, tokens, and trace metadata.
    """
    # 1. Core Workflow & Routing
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    sanitized_query: Optional[str]
    is_input_safe: Optional[bool]
    input_rejection_reason: Optional[str]
    intent: Optional[IntentType]
    intent_reasoning: Optional[str]
    next_step: Optional[str]

    # 2. Modular Sub-Pipeline Outputs (Structured & Compact)
    routing_output: Optional[Dict[str, Any]]
    sql_output: Optional[Dict[str, Any]]
    rag_output: Optional[Dict[str, Any]]
    judge_output: Optional[JudgeEvaluation]

    # 3. Pipeline Specific Fields (Direct access & backward compatibility)
    db_schema_context: Optional[str]
    generated_sql: Optional[str]
    is_sql_valid: Optional[bool]
    sql_validation_notes: Optional[str]
    sql_data: Optional[List[Dict[str, Any]]]
    sql_row_count: Optional[int]

    retrieved_chunks: Optional[List[RetrievedChunk]]
    citations: Optional[List[str]]
    candidate_answer: Optional[str]
    judge_evaluation: Optional[JudgeEvaluation]

    # 4. Final Delivery, Session Memory & Observability
    final_answer: Optional[str]
    chat_history: Optional[List[Dict[str, Any]]]
    retry_count: int
    errors: Annotated[List[str], operator.add]
    metadata: Optional[Dict[str, Any]]
