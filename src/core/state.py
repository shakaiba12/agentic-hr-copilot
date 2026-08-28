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
    RISKY = "RISKY"
    CASUAL = "CASUAL"
    UNKNOWN = "UNKNOWN"


class RetrievedChunk(TypedDict):
    """Metadata and text content of a retrieved document passage."""
    text: str
    source: str
    doc_type: str
    page: Optional[int]
    score: Optional[float]


class AgentState(TypedDict):
    """
    Global state graph schema for PeopleQuery AI multi-agent orchestration.
    """
    # User Input
    query: str
    
    # Routing & Intent
    intent: Optional[IntentType]
    intent_reasoning: Optional[str]
    
    # Message History (appends with operator.add)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # SQL Pipeline State
    db_schema_context: Optional[str]
    generated_sql: Optional[str]
    is_sql_valid: Optional[bool]
    sql_validation_notes: Optional[str]
    sql_data: Optional[List[Dict[str, Any]]]
    sql_row_count: Optional[int]
    
    # RAG Pipeline State
    retrieved_chunks: Optional[List[RetrievedChunk]]
    citations: Optional[List[str]]
    
    # Human-in-the-Loop & Safety Gate
    requires_human_approval: Optional[bool]
    approval_reason: Optional[str]
    approval_status: Optional[str]  # "pending" | "approved" | "rejected"
    
    # Output & Evaluation
    final_answer: Optional[str]
    evaluation_score: Optional[float]
    evaluation_feedback: Optional[str]
    
    # Observability & Errors (appends with operator.add)
    errors: Annotated[List[str], operator.add]
    metadata: Optional[Dict[str, Any]]
