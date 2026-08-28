import pytest
from src.core.config import Settings, get_settings
from src.core.state import AgentState, IntentType
from src.core.llm import get_llm


def test_settings_initialization():
    """Test that Settings loads default values correctly."""
    settings = get_settings()
    assert settings.PROJECT_NAME == "PeopleQuery AI"
    assert settings.DEFAULT_PROVIDER in ["gemini", "openai", "groq"]
    assert settings.DATABASE_URL.startswith("sqlite")
    assert settings.DB_QUERY_TIMEOUT_SECONDS > 0


def test_agent_state_schema():
    """Test that AgentState can be constructed with expected keys and types."""
    state: AgentState = {
        "query": "How many employees are in Sales?",
        "intent": IntentType.SQL_ONLY,
        "intent_reasoning": "Query asks for headcount in a department.",
        "messages": [],
        "db_schema_context": "TABLE employees (id INT, department VARCHAR)",
        "generated_sql": "SELECT COUNT(*) FROM employees WHERE department = 'Sales'",
        "is_sql_valid": True,
        "sql_validation_notes": "Valid SELECT query",
        "sql_data": [{"count": 42}],
        "sql_row_count": 1,
        "retrieved_chunks": None,
        "citations": None,
        "requires_human_approval": False,
        "approval_reason": None,
        "approval_status": None,
        "final_answer": "There are 42 employees in Sales.",
        "evaluation_score": 1.0,
        "evaluation_feedback": "Accurate response.",
        "errors": [],
        "metadata": {"test": True},
    }

    assert state["query"] == "How many employees are in Sales?"
    assert state["intent"] == IntentType.SQL_ONLY
    assert state["is_sql_valid"] is True
    assert len(state["sql_data"]) == 1


def test_intent_type_enum():
    """Test that all required intent types exist."""
    assert IntentType.SQL_ONLY == "SQL_ONLY"
    assert IntentType.RAG_ONLY == "RAG_ONLY"
    assert IntentType.HYBRID == "HYBRID"
    assert IntentType.RISKY == "RISKY"
    assert IntentType.CASUAL == "CASUAL"
    assert IntentType.UNKNOWN == "UNKNOWN"


def test_llm_factory_error_on_missing_key(monkeypatch):
    """Test that get_llm raises an informative error when API key is missing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_llm(provider="gemini")
    
    get_settings.cache_clear()
