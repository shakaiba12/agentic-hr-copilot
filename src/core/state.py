from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class IntentType(str, Enum):
    """Classification intent types for query routing."""
    SQL_ONLY = "SQL_ONLY"
    RAG_ONLY = "RAG_ONLY"
    HYBRID = "HYBRID"
    CASUAL = "CASUAL"
    UNKNOWN = "UNKNOWN"


class RetrievedChunk(TypedDict):
    """Metadata and text content of a retrieved document passage."""
    text: str
    source: str
    doc_type: str
    section: Optional[str]
    page: Optional[int]
    score: Optional[float]


class JudgeEvaluation(TypedDict):
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


class AgentState(TypedDict):
    """
    Global state graph schema for PeopleQuery AI Single AI Orchestrator.
    Tracks state across Guardrails -> Orchestrator -> Pipelines -> LLM Judge -> Delivery.
    """
    # 1. User Input & Input Guardrails
    query: str
    sanitized_query: Optional[str]
    is_input_safe: Optional[bool]
    input_rejection_reason: Optional[str]

    # 2. Query Classification & Routing
    intent: Optional[IntentType]
    intent_reasoning: Optional[str]

    # 3. Message History
    messages: Annotated[List[BaseMessage], operator.add]

    # 4. SQL Pipeline State
    db_schema_context: Optional[str]
    generated_sql: Optional[str]
    is_sql_valid: Optional[bool]
    sql_validation_notes: Optional[str]
    sql_data: Optional[List[Dict[str, Any]]]
    sql_row_count: Optional[int]

    # 5. RAG Pipeline State
    retrieved_chunks: Optional[List[RetrievedChunk]]
    citations: Optional[List[str]]

    # 6. Candidate Answer Generation
    candidate_answer: Optional[str]

    # 7. LLM Judge / Evaluator & Self-Correction Gate
    judge_evaluation: Optional[JudgeEvaluation]
    retry_count: int

    # 8. Output Guardrails & Final Response
    final_answer: Optional[str]

    # 9. Observability, Latency & Error Tracking
    errors: Annotated[List[str], operator.add]
    metadata: Optional[Dict[str, Any]]

