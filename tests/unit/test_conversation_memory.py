"""
Unit & Integration tests for PeopleQuery AI Conversation Memory, Session Isolation,
and Follow-Up Contextual Handling.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.server import app
from src.core.memory import ConversationStore, get_conversation_store
from src.core.orchestrator import MasterOrchestrator, OrchestratorResponse
from src.rag.router import QueryRouter, RouteCategory, RouteDecision
from src.sql.pipeline import SQLPipelineResult
from src.rag.pipeline import RAGPipelineResult


@pytest.fixture
def memory_store(tmp_path):
    """Fixture providing an isolated SQLite conversation store."""
    db_file = tmp_path / "test_memory.sqlite"
    return ConversationStore(db_path=str(db_file))


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# =========================================================================
# 1. CONVERSATION STORE & PERSISTENCE TESTS
# =========================================================================

class TestConversationStore:
    def test_create_conversation_and_defaults(self, memory_store: ConversationStore):
        conv = memory_store.create_conversation()
        assert conv.id is not None
        assert conv.title == "New Conversation"
        assert conv.message_count == 0

        fetched = memory_store.get_conversation(conv.id)
        assert fetched is not None
        assert fetched.id == conv.id
        assert fetched.title == "New Conversation"

    def test_add_and_retrieve_messages(self, memory_store: ConversationStore):
        conv = memory_store.create_conversation(title="Austin Query")
        
        # Add user message
        msg1 = memory_store.add_message(
            conversation_id=conv.id,
            role="user",
            content="How many employees work in Austin?",
        )
        assert msg1.id is not None
        assert msg1.role == "user"
        assert msg1.content == "How many employees work in Austin?"

        # Add assistant message with metadata
        meta = {"allowed": True, "source": "sql", "rows": [{"employee_count": 7}]}
        msg2 = memory_store.add_message(
            conversation_id=conv.id,
            role="assistant",
            content="Found 7 employees in Austin.",
            source="sql",
            category="DATA_QUERY",
            metadata=meta,
        )
        assert msg2.source == "sql"
        assert msg2.metadata == meta

        # Retrieve messages
        messages = memory_store.get_messages(conv.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Found 7 employees in Austin."

        # Verify conversation message count
        conv_updated = memory_store.get_conversation(conv.id)
        assert conv_updated.message_count == 2

    def test_auto_title_generation_from_first_user_message(self, memory_store: ConversationStore):
        conv = memory_store.create_conversation()
        assert conv.title == "New Conversation"

        memory_store.add_message(
            conversation_id=conv.id,
            role="user",
            content="Which employees were hired after January 1, 2024?",
        )

        conv_updated = memory_store.get_conversation(conv.id)
        assert "Which employees were hired" in conv_updated.title

    def test_delete_conversation_and_cascade(self, memory_store: ConversationStore):
        conv = memory_store.create_conversation()
        memory_store.add_message(conv.id, "user", "Hello")
        memory_store.add_message(conv.id, "assistant", "Hi there")

        assert len(memory_store.get_messages(conv.id)) == 2

        deleted = memory_store.delete_conversation(conv.id)
        assert deleted is True
        assert memory_store.get_conversation(conv.id) is None
        assert len(memory_store.get_messages(conv.id)) == 0

    def test_list_conversations_order(self, memory_store: ConversationStore):
        conv1 = memory_store.create_conversation(title="First")
        conv2 = memory_store.create_conversation(title="Second")

        # Update conv1 by adding message
        memory_store.add_message(conv1.id, "user", "Message to first")

        convs = memory_store.list_conversations()
        assert len(convs) >= 2
        # conv1 was updated last, should appear first
        assert convs[0].id == conv1.id


# =========================================================================
# 2. SESSION ISOLATION & NO LEAKAGE TESTS
# =========================================================================

class TestConversationIsolation:
    def test_conversation_isolation_no_data_leak(self, memory_store: ConversationStore):
        """Messages in Conversation A must never appear in Conversation B."""
        conv_a = memory_store.create_conversation(title="Thread A")
        conv_b = memory_store.create_conversation(title="Thread B")

        memory_store.add_message(conv_a.id, "user", "How many employees work in Austin?")
        memory_store.add_message(conv_a.id, "assistant", "Found 7 employees in Austin.", source="sql")

        memory_store.add_message(conv_b.id, "user", "What is the parental leave policy?")
        memory_store.add_message(conv_b.id, "assistant", "16 weeks paid parental leave.", source="rag")

        history_a = memory_store.get_history_for_orchestrator(conv_a.id)
        history_b = memory_store.get_history_for_orchestrator(conv_b.id)

        assert len(history_a) == 2
        assert len(history_b) == 2

        assert "Austin" in history_a[0]["content"]
        assert "parental leave" not in history_a[0]["content"]

        assert "parental leave" in history_b[0]["content"]
        assert "Austin" not in history_b[0]["content"]


# =========================================================================
# 3. ROUTER FOLLOW-UP & GENERAL INTENT HANDLING
# =========================================================================

class TestFollowUpContextRouting:
    def test_followup_question_receives_previous_sql_context(self):
        router = QueryRouter()
        history = [
            {"role": "user", "content": "How many employees work in Austin?", "category": RouteCategory.DATA_QUERY},
            {"role": "assistant", "content": "Found 7 employees in Austin.", "source": "sql"},
        ]

        decision = router.route("What about Engineering?", history=history)
        assert decision.category == RouteCategory.DATA_QUERY
        assert decision.allowed is True
        assert decision.target == "data_specialist"

    def test_followup_question_receives_previous_rag_context(self):
        router = QueryRouter()
        history = [
            {"role": "user", "content": "What is the parental leave policy?", "category": RouteCategory.RAG_KNOWLEDGE},
            {"role": "assistant", "content": "16 weeks paid parental leave.", "source": "rag"},
        ]

        decision = router.route("How long is it?", history=history)
        assert decision.category == RouteCategory.RAG_KNOWLEDGE
        assert decision.allowed is True
        assert decision.target == "rag"

    def test_casual_greetings_handled_without_sql_or_rag(self):
        router = QueryRouter()
        for greeting in ["Hi", "Hello", "Hey there", "Good morning", "Thanks!"]:
            decision = router.route(greeting)
            assert decision.category == RouteCategory.CASUAL
            assert decision.allowed is True
            assert decision.target == "general"


# =========================================================================
# 4. API CONVERSATION ENDPOINTS & CHAT MEMORY INTEGRATION
# =========================================================================

class TestApiConversationEndpoints:
    def test_conversations_crud_api(self, api_client):
        # 1. Create conversation
        res = api_client.post("/api/conversations", json={"title": "Test Chat"})
        assert res.status_code == 200
        data = res.json()
        conv_id = data["id"]
        assert data["title"] == "Test Chat"

        # 2. List conversations
        res_list = api_client.get("/api/conversations")
        assert res_list.status_code == 200
        conv_ids = [c["id"] for c in res_list.json()]
        assert conv_id in conv_ids

        # 3. Get conversation detail
        res_detail = api_client.get(f"/api/conversations/{conv_id}")
        assert res_detail.status_code == 200
        assert res_detail.json()["id"] == conv_id

        # 4. Delete conversation
        res_del = api_client.delete(f"/api/conversations/{conv_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

    @patch("src.api.routes.get_orchestrator")
    def test_chat_endpoint_persists_messages(self, mock_get_orch, api_client):
        mock_orch = MagicMock()
        mock_get_orch.return_value = mock_orch

        decision = RouteDecision(
            category=RouteCategory.DATA_QUERY,
            target="sql",
            allowed=True,
            confidence=1.0,
            reason="Austin employee lookup",
        )
        sql_res = SQLPipelineResult(
            success=True,
            query="How many employees work in Austin?",
            generated_sql="SELECT COUNT(*) FROM employees",
            rows=[{"count": 7}],
            row_count=1,
            message="Found 7 employees.",
        )
        mock_orch.process_query.return_value = OrchestratorResponse(
            decision=decision,
            source="sql",
            response="Found 7 employees.",
            allowed=True,
            sql_result=sql_res,
        )

        # Send chat message without explicit conversation_id
        res = api_client.post("/api/chat", json={"query": "How many employees work in Austin?"})
        assert res.status_code == 200
        data = res.json()
        assert "conversation_id" in data
        conv_id = data["conversation_id"]

        # Fetch conversation details to verify messages were persisted in database
        res_detail = api_client.get(f"/api/conversations/{conv_id}")
        assert res_detail.status_code == 200
        messages = res_detail.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "How many employees work in Austin?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Found 7 employees."
