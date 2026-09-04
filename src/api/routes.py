"""
PeopleQuery AI - API Endpoints
Provides REST and SSE streaming endpoints for the PeopleQuery AI Copilot with Conversation Memory.
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
from src.core.memory import ConversationStore, get_conversation_store
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
    conversation_id: Optional[str] = Field(default=None, description="Active conversation identifier")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional explicit conversation turns")


class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(default="New Conversation", description="Optional conversation title")


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageDetail(BaseModel):
    id: str
    role: str
    content: str
    source: Optional[str] = None
    category: Optional[str] = None
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageDetail] = Field(default_factory=list)


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
    conversation_id: str
    conversation_title: Optional[str] = None
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


# =========================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =========================================================================

@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(limit: int = 50) -> List[ConversationResponse]:
    """List recent conversations sorted by last update."""
    store = get_conversation_store()
    convs = store.list_conversations(limit=limit)
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in convs
    ]


@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(payload: Optional[CreateConversationRequest] = None) -> ConversationResponse:
    """Create a new conversation session."""
    store = get_conversation_store()
    title = payload.title if payload and payload.title else "New Conversation"
    conv = store.create_conversation(title=title)
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    """Retrieve conversation details and full message history."""
    store = get_conversation_store()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    msgs = store.get_messages(conversation_id)
    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageDetail(
                id=m.id,
                role=m.role,
                content=m.content,
                source=m.source,
                category=m.category,
                created_at=m.created_at,
                metadata=m.metadata,
            )
            for m in msgs
        ],
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """Delete a conversation and its messages."""
    store = get_conversation_store()
    success = store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "id": conversation_id}


# =========================================================================
# CHAT ENDPOINTS WITH MEMORY INTEGRATION
# =========================================================================

@router.post("/chat", response_model=ChatResponse)
def handle_chat(payload: ChatRequest) -> ChatResponse:
    """Process a user query through the Master Orchestrator with conversation memory."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    store = get_conversation_store()
    settings = get_settings()

    # 1. Resolve conversation
    conv = store.get_or_create_conversation(payload.conversation_id)
    conversation_id = conv.id

    # 2. Resolve history context
    if payload.history is not None and len(payload.history) > 0:
        history_context = payload.history
    else:
        history_context = store.get_history_for_orchestrator(
            conversation_id=conversation_id,
            max_turns=settings.MAX_CONVERSATION_HISTORY,
        )

    # 3. Store user message in memory
    store.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    orchestrator = get_orchestrator()

    try:
        res: OrchestratorResponse = orchestrator.process_query(query, history=history_context)
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

    # 4. Save assistant response into memory
    meta_to_save: Dict[str, Any] = {
        "allowed": res.allowed,
        "decision": decision_payload.model_dump(),
    }
    if rag_payload:
        meta_to_save["rag_result"] = rag_payload.model_dump()
    if sql_payload:
        meta_to_save["sql_result"] = sql_payload.model_dump()

    store.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=res.response,
        source=res.source,
        category=res.decision.category.value,
        metadata=meta_to_save,
    )

    # Refresh conversation info
    updated_conv = store.get_conversation(conversation_id)
    conv_title = updated_conv.title if updated_conv else conv.title

    return ChatResponse(
        query=query,
        response=res.response,
        source=res.source,
        allowed=res.allowed,
        conversation_id=conversation_id,
        conversation_title=conv_title,
        decision=decision_payload,
        rag_result=rag_payload,
        sql_result=sql_payload,
        error=res.error,
    )


@router.post("/chat/stream")
async def handle_chat_stream(payload: ChatRequest):
    """
    SSE stream endpoint for real-time progressive response delivery with memory persistence.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    store = get_conversation_store()
    settings = get_settings()

    # Resolve conversation and history
    conv = store.get_or_create_conversation(payload.conversation_id)
    conversation_id = conv.id

    if payload.history is not None and len(payload.history) > 0:
        history_context = payload.history
    else:
        history_context = store.get_history_for_orchestrator(
            conversation_id=conversation_id,
            max_turns=settings.MAX_CONVERSATION_HISTORY,
        )

    # Store user message
    store.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    orchestrator = get_orchestrator()

    async def event_generator():
        try:
            # Yield initial metadata
            res: OrchestratorResponse = await asyncio.to_thread(
                orchestrator.process_query, query, history=history_context
            )

            meta_data = {
                "conversation_id": conversation_id,
                "category": res.decision.category.value,
                "target": res.decision.target,
                "source": res.source,
                "allowed": res.allowed,
                "confidence": res.decision.confidence,
                "reason": res.decision.reason,
            }

            rag_meta = None
            if res.rag_result:
                rag_meta = {
                    "success": res.rag_result.success,
                    "sources": res.rag_result.sources,
                    "chunks_count": res.rag_result.chunks_count,
                    "grounded": res.rag_result.grounded,
                }
                meta_data["sources"] = res.rag_result.sources
                meta_data["chunks_count"] = res.rag_result.chunks_count
                meta_data["grounded"] = res.rag_result.grounded

            sql_meta = None
            if res.sql_result:
                was_trunc = (
                    res.sql_result.execution.was_truncated
                    if res.sql_result.execution
                    else False
                )
                sql_meta = {
                    "success": res.sql_result.success,
                    "generated_sql": res.sql_result.generated_sql,
                    "rows": res.sql_result.rows,
                    "row_count": res.sql_result.row_count,
                    "was_truncated": was_trunc,
                }
                meta_data["generated_sql"] = res.sql_result.generated_sql
                meta_data["rows"] = res.sql_result.rows
                meta_data["row_count"] = res.sql_result.row_count
                meta_data["was_truncated"] = was_trunc

            yield f"event: meta\ndata: {json.dumps(meta_data)}\n\n"

            # Stream response in natural word-sized tokens for smooth rendering
            full_text = res.response or ""
            words = full_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"event: token\ndata: {json.dumps({'delta': token})}\n\n"
                await asyncio.sleep(0.012)

            # Persist completed assistant message
            save_meta: Dict[str, Any] = {
                "allowed": res.allowed,
                "decision": {
                    "category": res.decision.category.value,
                    "target": res.decision.target,
                    "allowed": res.decision.allowed,
                    "confidence": res.decision.confidence,
                    "reason": res.decision.reason,
                },
            }
            if rag_meta:
                save_meta["rag_result"] = rag_meta
            if sql_meta:
                save_meta["sql_result"] = sql_meta

            store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=res.response,
                source=res.source,
                category=res.decision.category.value,
                metadata=save_meta,
            )

            updated_conv = store.get_conversation(conversation_id)
            conv_title = updated_conv.title if updated_conv else conv.title

            yield f"event: done\ndata: {json.dumps({'status': 'complete', 'conversation_id': conversation_id, 'conversation_title': conv_title})}\n\n"

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
