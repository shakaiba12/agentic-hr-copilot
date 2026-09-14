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
    doc_type: Optional[str]
    section: Optional[str]
    page: Optional[int]
    score: Optional[float]


class JudgeEvaluation(TypedDict, total=False):
    """Structured output from the LLM Judge / Evaluator."""
    decision: str
    score: float
    correctness: float
    relevance: float
    faithfulness: float
    completeness: float
    safety: float
    issues: List[str]
    feedback: Optional[str]


class AgentState(TypedDict, total=False):
    """
    Shared Master Orchestrator State for the Agentic HR Copilot StateGraph.

    1. Input & Safety Guardrail:
       - messages: LangGraph conversational message history with add_messages reducer.
       - query: Current raw user question.
       - sanitized_query: Normalized and sanitized query string.
       - is_input_safe: Flag indicating whether input passed safety guardrails.
       - input_rejection_reason: Explanation if input guardrail rejected the query.

    2. Intent & Routing:
       - intent: Categorized intent (SQL_ONLY, RAG_ONLY, HYBRID, CASUAL, UNKNOWN).
       - intent_reasoning: Rationale provided by the Query Router.
       - route_decision: Full structured RouteDecision object.
       - routing_output: Summary dictionary of routing metadata.

    3. Subgraph Pipeline Execution & Bridging:
       - sql_output: Structured dictionary from the SQL Agent Subgraph.
       - sql_result: Full SQLPipelineResult object.
       - rag_output: Structured dictionary from the RAG Agent Subgraph.
       - rag_result: Full RAGPipelineResult object.
       - generated_sql: Validated SQL query string (bridged for LLM Judge).
       - sql_data: Execution result rows (bridged for LLM Judge).
       - sql_row_count: Number of rows returned by SQL query.
       - is_sql_valid: SQL validation status boolean.
       - retrieved_chunks: Retrieved document chunks (bridged for Judge/Regeneration).
       - citations: Citation document references (bridged for Output Guardrail).

    4. Synthesis, Judge & Regeneration:
       - candidate_answer: Intermediate synthesized response.
       - judge_decision: Structured JudgeDecision evaluation object.
       - judge_output: Summary dictionary from the LLM Judge node.
       - judge_evaluation: Structured JudgeEvaluation score breakdown.
       - retry_count: Number of bounded self-correction regeneration retries.

    5. Final Output & Observability:
       - source: Pipeline handler identifier ('sql', 'rag', 'hybrid', 'master').
       - output_guardrail_result: Sanitization and PII masking results.
       - final_answer: Verified final response delivered to the user.
       - chat_history: Multi-turn conversation history list.
       - errors: Accumulated execution errors with operator.add reducer.
       - metadata: Request ID, session ID, latency, and observability metadata.
    """
    # 1. Input & Safety Guardrail
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    sanitized_query: Optional[str]
    is_input_safe: Optional[bool]
    input_rejection_reason: Optional[str]

    # 2. Intent & Routing
    intent: Optional[IntentType]
    intent_reasoning: Optional[str]
    route_decision: Optional[Any]
    routing_output: Optional[Dict[str, Any]]

    # 3. Subgraph Pipeline Outputs & Bridging Fields
    sql_output: Optional[Dict[str, Any]]
    sql_result: Optional[Any]
    rag_output: Optional[Dict[str, Any]]
    rag_result: Optional[Any]

    generated_sql: Optional[str]
    sql_data: Optional[List[Dict[str, Any]]]
    sql_row_count: Optional[int]
    is_sql_valid: Optional[bool]
    retrieved_chunks: Optional[List[RetrievedChunk]]
    citations: Optional[List[str]]

    # Compatibility / Test fixtures
    db_schema_context: Optional[str]
    sql_validation_notes: Optional[str]

    # 4. Synthesis, Judge & Regeneration
    candidate_answer: Optional[str]
    judge_decision: Optional[Any]
    judge_output: Optional[Dict[str, Any]]
    judge_evaluation: Optional[JudgeEvaluation]
    retry_count: int

    # 5. Final Output & Observability
    source: Optional[str]
    output_guardrail_result: Optional[Any]
    final_answer: Optional[str]
    chat_history: Optional[List[Dict[str, Any]]]
    errors: Annotated[List[str], operator.add]
    metadata: Optional[Dict[str, Any]]


class SQLState(TypedDict, total=False):
    """Execution state for the SQL Agent Subgraph."""
    query: str
    sanitized_query: Optional[str]
    chat_history: Optional[List[Dict[str, Any]]]
    db_schema_context: Optional[str]
    generated_sql: Optional[str]
    is_sql_valid: Optional[bool]
    sql_validation_notes: Optional[str]
    sql_data: Optional[List[Dict[str, Any]]]
    sql_row_count: Optional[int]
    sql_retry_count: int
    sql_error: Optional[str]
    sql_output: Optional[Dict[str, Any]]
    sql_result: Optional[Any]
    candidate_answer: Optional[str]
    judge_decision: Optional[Any]
    errors: Annotated[List[str], operator.add]


class RAGState(TypedDict, total=False):
    """Execution state for the RAG Agent Subgraph."""
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
