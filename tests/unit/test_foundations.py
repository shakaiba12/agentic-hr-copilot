import os
import sqlite3
import pytest
from src.core.config import Settings, get_settings
from src.core.state import AgentState, IntentType, JudgeEvaluation, RetrievedChunk
from src.core.llm import get_llm
from src.core.observability import time_execution


def test_settings_initialization():
    """Test that Settings loads default values correctly."""
    settings = get_settings()
    assert "PeopleQuery" in settings.PROJECT_NAME
    assert settings.DEFAULT_PROVIDER in ["gemini", "openai", "groq", "ollama"]
    assert settings.DATABASE_URL.startswith("sqlite")
    assert settings.DB_QUERY_TIMEOUT_SECONDS > 0
    assert settings.MAX_EVALUATION_RETRIES == 2
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert str(settings.CHROMA_PERSIST_DIR).endswith("chroma_db")


def test_agent_state_schema():
    """Test that AgentState can be constructed with all typed fields for Single AI Orchestrator."""
    judge_eval: JudgeEvaluation = {
        "decision": "PASS",
        "score": 0.95,
        "correctness": 0.96,
        "relevance": 0.95,
        "faithfulness": 0.98,
        "completeness": 0.92,
        "safety": 1.0,
        "issues": [],
        "feedback": "Grounded and accurate.",
    }

    chunk: RetrievedChunk = {
        "text": "16 weeks of paid maternity leave after 12 months continuous service.",
        "source": "leave_policy.md",
        "doc_type": "markdown",
        "section": "Parental Leave",
        "page": 1,
        "score": 0.89,
    }

    state: AgentState = {
        "query": "How many employees are eligible for maternity leave?",
        "sanitized_query": "How many employees are eligible for maternity leave?",
        "is_input_safe": True,
        "input_rejection_reason": None,
        "intent": IntentType.HYBRID,
        "intent_reasoning": "Requires policy eligibility rules and employee database records.",
        "messages": [],
        "db_schema_context": "TABLE employees (id, hire_date, gender...)",
        "generated_sql": "SELECT COUNT(*) FROM employees WHERE gender = 'Female' AND hire_date <= date('now', '-1 year')",
        "is_sql_valid": True,
        "sql_validation_notes": "Safe SELECT query",
        "sql_data": [{"count": 8}],
        "sql_row_count": 1,
        "retrieved_chunks": [chunk],
        "citations": ["leave_policy.md # Parental Leave"],
        "candidate_answer": "8 employees currently satisfy the 12-month service requirement for maternity leave.",
        "judge_evaluation": judge_eval,
        "retry_count": 0,
        "final_answer": "8 employees currently satisfy the 12-month service requirement for maternity leave.",
        "errors": [],
        "metadata": {"execution_time_ms": 120},
    }

    assert state["query"].startswith("How many")
    assert state["intent"] == IntentType.HYBRID
    assert state["is_sql_valid"] is True
    assert state["judge_evaluation"]["decision"] == "PASS"
    assert len(state["retrieved_chunks"]) == 1


def test_intent_type_enum():
    """Test that all required intent types exist."""
    assert IntentType.SQL_ONLY == "SQL_ONLY"
    assert IntentType.RAG_ONLY == "RAG_ONLY"
    assert IntentType.HYBRID == "HYBRID"
    assert IntentType.CASUAL == "CASUAL"
    assert IntentType.UNKNOWN == "UNKNOWN"


def test_llm_factory_error_on_missing_key(monkeypatch):
    """Test that get_llm raises an informative error when API key is missing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_llm(provider="gemini")

    get_settings.cache_clear()


def test_llm_factory_ollama():
    """Test that get_llm instantiates ChatOpenAI instance for Ollama."""
    llm = get_llm(provider="ollama", model_name="deepseek-r1:latest")
    assert llm is not None
    assert "11434" in str(llm.openai_api_base) or "11434" in str(getattr(llm, "base_url", ""))


def test_database_seeded_correctly():
    """Verify that SQLite seed database exists and contains expected tables and rows."""
    settings = get_settings()
    db_file = settings.DATABASE_URL.replace("sqlite:///", "")
    assert os.path.exists(db_file), f"Database file not found at {db_file}"

    conn = sqlite3.connect(db_file)

    # Check tables
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "departments" in tables
    assert "positions" in tables
    assert "employees" in tables
    assert "leaves" in tables
    assert "benefits" in tables
    assert "performance_reviews" in tables

    # Check employee count
    emp_count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    assert emp_count >= 20

    # Check department count
    dept_count = conn.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
    assert dept_count >= 6

    conn.close()


def test_company_docs_exist():
    """Verify that HR policy documents exist in company_docs directory."""
    settings = get_settings()
    docs_dir = settings.DOCS_DIR
    assert docs_dir.exists()

    expected_docs = [
        "leave_policy.md",
        "remote_work_policy.md",
        "employee_handbook.md",
        "compensation_benefits.md",
    ]
    for doc_name in expected_docs:
        doc_path = docs_dir / doc_name
        assert doc_path.exists(), f"Policy document {doc_name} is missing"
        assert len(doc_path.read_text(encoding="utf-8")) > 100


def test_observability_timing_helper():
    """Verify that time_execution context manager measures duration."""
    meta = {}
    with time_execution("test_step", meta):
        _ = sum(i * i for i in range(1000))

    assert "timings" in meta
    assert "test_step" in meta["timings"]
    assert meta["timings"]["test_step"] >= 0.0
