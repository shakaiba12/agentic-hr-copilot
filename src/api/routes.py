"""
PeopleQuery AI - API Endpoints
Provides REST and SSE streaming endpoints for the PeopleQuery AI Copilot.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.orchestrator import MasterOrchestrator, OrchestratorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Copilot"])

# Lazy singleton orchestrator
_orchestrator: Optional[MasterOrchestrator] = None


def get_orchestrator() -> MasterOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        settings = get_settings()
        _orchestrator = MasterOrchestrator(settings=settings)
    return _orchestrator


class ChatMessage(BaseModel):
    role: str
    content: str
    category: Optional[str] = None
    source: Optional[str] = None


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or workforce analytics query")
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Recent conversation turns")


class RouteDecisionPayload(BaseModel):
    category: str
    target: Optional[str]
    allowed: bool
    confidence: float
    reason: str


class RAGResultPayload(BaseModel):
    success: bool
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    chunks_count: int = 0
    grounded: bool = True
    error: Optional[str] = None


class SQLResultPayload(BaseModel):
    success: bool
    generated_sql: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    message: str = ""
    error: Optional[str] = None
    was_truncated: bool = False


class ChatResponse(BaseModel):
    query: str
    response: str
    source: str
    allowed: bool
    decision: RouteDecisionPayload
    rag_result: Optional[RAGResultPayload] = None
    sql_result: Optional[SQLResultPayload] = None
    error: Optional[str] = None


@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Return backend status and active configuration."""
    settings = get_settings()
    docs_dir = Path("company_docs")
    doc_files = list(docs_dir.glob("*.md")) if docs_dir.exists() else []

    return {
        "status": "healthy",
        "app": "PeopleQuery AI",
        "environment": settings.APP_ENV,
        "provider": settings.DEFAULT_PROVIDER,
        "model": settings.DEFAULT_MODEL,
        "database": settings.DATABASE_URL,
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "documents_count": len(doc_files),
    }


@router.get("/docs/list")
def list_documents() -> Dict[str, Any]:
    """List loaded HR policy documents from company_docs."""
    docs_dir = Path("company_docs")
    documents = []

    if docs_dir.exists():
        for f in sorted(docs_dir.glob("*.md")):
            # Compute friendly title from filename
            title = f.stem.replace("-", " ").replace("_", " ").title()
            documents.append({
                "filename": f.name,
                "title": title,
                "size_bytes": f.stat().st_size,
            })

    return {
        "documents": documents,
        "count": len(documents),
    }


@router.post("/chat", response_model=ChatResponse)
def handle_chat(payload: ChatRequest) -> ChatResponse:
    """Process a user query through the Master Orchestrator."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    orchestrator = get_orchestrator()

    try:
        res: OrchestratorResponse = orchestrator.process_query(query, history=payload.history)
    except Exception as e:
        logger.exception("Error processing orchestrator query")
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")

    decision_payload = RouteDecisionPayload(
        category=res.decision.category.value,
        target=res.decision.target,
        allowed=res.decision.allowed,
        confidence=res.decision.confidence,
        reason=res.decision.reason,
    )

    rag_payload = None
    if res.rag_result:
        rag_payload = RAGResultPayload(
            success=res.rag_result.success,
            sources=res.rag_result.sources,
            chunks_count=res.rag_result.chunks_count,
            grounded=res.rag_result.grounded,
            error=res.rag_result.error,
        )

    sql_payload = None
    if res.sql_result:
        was_truncated = False
        if res.sql_result.execution:
            was_truncated = res.sql_result.execution.was_truncated

        sql_payload = SQLResultPayload(
            success=res.sql_result.success,
            generated_sql=res.sql_result.generated_sql,
            rows=res.sql_result.rows,
            row_count=res.sql_result.row_count,
            message=res.sql_result.message,
            error=res.sql_result.error,
            was_truncated=was_truncated,
        )

    return ChatResponse(
        query=query,
        response=res.response,
        source=res.source,
        allowed=res.allowed,
        decision=decision_payload,
        rag_result=rag_payload,
        sql_result=sql_payload,
        error=res.error,
    )


@router.post("/chat/stream")
async def handle_chat_stream(payload: ChatRequest):
    """
    SSE stream endpoint for real-time progressive response delivery.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    orchestrator = get_orchestrator()

    async def event_generator():
        try:
            # Yield initial metadata
            res: OrchestratorResponse = await asyncio.to_thread(
                orchestrator.process_query, query, history=payload.history
            )

            meta_data = {
                "category": res.decision.category.value,
                "target": res.decision.target,
                "source": res.source,
                "allowed": res.allowed,
                "confidence": res.decision.confidence,
                "reason": res.decision.reason,
            }

            if res.rag_result:
                meta_data["sources"] = res.rag_result.sources
                meta_data["chunks_count"] = res.rag_result.chunks_count
                meta_data["grounded"] = res.rag_result.grounded

            if res.sql_result:
                meta_data["generated_sql"] = res.sql_result.generated_sql
                meta_data["rows"] = res.sql_result.rows
                meta_data["row_count"] = res.sql_result.row_count
                meta_data["was_truncated"] = (
                    res.sql_result.execution.was_truncated
                    if res.sql_result.execution
                    else False
                )

            yield f"event: meta\ndata: {json.dumps(meta_data)}\n\n"

            # Stream response in natural word-sized tokens for smooth rendering
            full_text = res.response or ""
            words = full_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"event: token\ndata: {json.dumps({'delta': token})}\n\n"
                await asyncio.sleep(0.012)

            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            logger.exception("Stream error")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
