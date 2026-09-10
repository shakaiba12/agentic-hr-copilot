"""
Pydantic schemas for the LLM-as-a-Judge verification layer.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerdictStatus(str, Enum):
    """Status indicating how the answer was verified or processed."""
    VERIFIED = "VERIFIED"
    REJECTED_AND_REGENERATED = "REJECTED_AND_REGENERATED"
    JUDGE_UNAVAILABLE = "JUDGE_UNAVAILABLE"
    SKIPPED = "SKIPPED"


class JudgeDecision(BaseModel):
    """Structured decision output from the LLM Judge."""
    passed: bool = Field(
        description="Whether the answer passed all verification criteria."
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence/quality score between 0.0 and 1.0."
    )
    correctness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Factual correctness score between 0.0 and 1.0."
    )
    groundedness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Groundedness / faithfulness score between 0.0 and 1.0."
    )
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance score between 0.0 and 1.0."
    )
    hallucination: bool = Field(
        default=False,
        description="Whether hallucination or unsupported claims were detected."
    )
    needs_regeneration: bool = Field(
        default=False,
        description="Whether the answer requires regeneration."
    )
    grounded: bool = Field(
        default=True,
        description="Whether every important claim is grounded in retrieved documents/database results."
    )
    relevant: bool = Field(
        default=True,
        description="Whether the answer directly and accurately addresses the user's question."
    )
    complete: bool = Field(
        default=True,
        description="Whether the answer covers the core parts of the question."
    )
    contradiction: bool = Field(
        default=False,
        description="Whether the answer contradicts provided evidence or SQL records."
    )
    result_consistent: Optional[bool] = Field(
        default=None,
        description="For SQL: Whether the answer accurately and numerically reflects the SQL execution result."
    )
    query_relevant: Optional[bool] = Field(
        default=None,
        description="For SQL: Whether the SQL query is relevant to the question."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific flaws, hallucinations, or contradictions identified by the judge."
    )
    reason: str = Field(
        default="",
        description="Concise rationale explaining the judge's verdict."
    )
    verdict_status: VerdictStatus = Field(
        default=VerdictStatus.VERIFIED,
        description="Verification lifecycle status."
    )
    retries_attempted: int = Field(
        default=0,
        ge=0,
        description="Number of regeneration retries attempted before final verdict."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Execution latency in milliseconds for judge evaluation."
    )


class RAGJudgeInput(BaseModel):
    """Input payload provided to the RAG LLM Judge."""
    model_config = {"arbitrary_types_allowed": True}

    question: str
    answer: str
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    history: Optional[List[Dict[str, Any]]] = None

    def __init__(self, **data: Any):
        if "question" in data and not isinstance(data["question"], str):
            data["question"] = str(data["question"])
        if "answer" in data and not isinstance(data["answer"], str):
            data["answer"] = str(data["answer"])
        super().__init__(**data)


class SQLJudgeInput(BaseModel):
    """Input payload provided to the SQL LLM Judge."""
    model_config = {"arbitrary_types_allowed": True}

    question: str
    sql: str
    sql_result: Any
    answer: str
    history: Optional[List[Dict[str, Any]]] = None

    def __init__(self, **data: Any):
        if "question" in data and not isinstance(data["question"], str):
            data["question"] = str(data["question"])
        if "sql" in data and not isinstance(data["sql"], str):
            data["sql"] = str(data["sql"])
        if "answer" in data and not isinstance(data["answer"], str):
            data["answer"] = str(data["answer"])
        super().__init__(**data)
