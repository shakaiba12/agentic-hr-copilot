"""
Unit tests for PeopleQuery AI FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.server import app
from src.core.orchestrator import OrchestratorResponse
from src.rag.router import RouteCategory, RouteDecision
from src.sql.pipeline import SQLPipelineResult
from src.rag.pipeline import RAGPipelineResult


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "PeopleQuery AI"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "PeopleQuery AI"
    assert "provider" in data
    assert "model" in data
    assert "database" in data


def test_list_documents_endpoint(client):
    response = client.get("/api/docs/list")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert data["count"] >= 0


def test_chat_empty_query_rejected(client):
    response = client.post("/api/chat", json={"query": "   "})
    assert response.status_code == 400


@patch("src.api.routes.get_orchestrator")
def test_chat_sql_response(mock_get_orch, client):
    mock_orch = MagicMock()
    mock_get_orch.return_value = mock_orch

    decision = RouteDecision(
        category=RouteCategory.DATA_QUERY,
        target="sql",
        allowed=True,
        confidence=0.95,
        reason="Employee data lookup",
    )
    sql_res = SQLPipelineResult(
        success=True,
        query="How many employees work in Austin?",
        generated_sql="SELECT COUNT(*) as count FROM employees WHERE location LIKE '%Austin%'",
        rows=[{"count": 12}],
        row_count=1,
        message="Found 12 employees in Austin.",
    )
    mock_orch.process_query.return_value = OrchestratorResponse(
        decision=decision,
        source="sql",
        response="Found 12 employees in Austin.",
        allowed=True,
        sql_result=sql_res,
    )

    response = client.post("/api/chat", json={"query": "How many employees work in Austin?"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "sql"
    assert data["response"] == "Found 12 employees in Austin."
    assert data["sql_result"]["generated_sql"] is not None
    assert len(data["sql_result"]["rows"]) == 1
    assert data["sql_result"]["rows"][0]["count"] == 12


@patch("src.api.routes.get_orchestrator")
def test_chat_rag_response(mock_get_orch, client):
    mock_orch = MagicMock()
    mock_get_orch.return_value = mock_orch

    decision = RouteDecision(
        category=RouteCategory.RAG_KNOWLEDGE,
        target="rag",
        allowed=True,
        confidence=0.92,
        reason="Parental leave policy inquiry",
    )
    rag_res = RAGPipelineResult(
        success=True,
        query="What is the parental leave policy?",
        response="Employees receive 16 weeks of paid parental leave.",
        sources=[{"document_name": "parental-leave.md", "section_title": "Eligibility", "page_number": 1}],
        chunks_count=1,
        grounded=True,
    )
    mock_orch.process_query.return_value = OrchestratorResponse(
        decision=decision,
        source="rag",
        response="Employees receive 16 weeks of paid parental leave.",
        allowed=True,
        rag_result=rag_res,
    )

    response = client.post("/api/chat", json={"query": "What is the parental leave policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "rag"
    assert data["rag_result"]["success"] is True
    assert len(data["rag_result"]["sources"]) == 1
