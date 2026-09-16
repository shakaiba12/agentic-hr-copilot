"""
Routing definitions, category mappings, and fallback constants for PeopleQuery AI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from src.core.router import RouteCategory, RouteDecision
from src.core.state import IntentType

INTENT_ROUTES: Dict[IntentType, Union[str, List[str]]] = {
    IntentType.SQL_ONLY: "sql_agent",
    IntentType.RAG_ONLY: "rag_agent",
    IntentType.HYBRID: ["sql_agent", "rag_agent"],
    IntentType.CASUAL: "answer_synthesis",
    IntentType.UNKNOWN: "answer_synthesis",
}

CATEGORY_TO_INTENT: Dict[RouteCategory, Tuple[IntentType, str]] = {
    RouteCategory.DATA_QUERY: (IntentType.SQL_ONLY, "sql"),
    RouteCategory.RAG_KNOWLEDGE: (IntentType.RAG_ONLY, "rag"),
    RouteCategory.HYBRID: (IntentType.HYBRID, "hybrid"),
    RouteCategory.CASUAL: (IntentType.CASUAL, "master"),
    RouteCategory.GENERAL: (IntentType.CASUAL, "master"),
    RouteCategory.OUT_OF_SCOPE: (IntentType.CASUAL, "master"),
}

INTENT_FALLBACK_ROUTES: Dict[Optional[IntentType], Tuple[RouteCategory, Optional[str]]] = {
    IntentType.SQL_ONLY: (RouteCategory.DATA_QUERY, "data_specialist"),
    IntentType.RAG_ONLY: (RouteCategory.RAG_KNOWLEDGE, "rag"),
    IntentType.HYBRID: (RouteCategory.HYBRID, "hybrid"),
    IntentType.CASUAL: (RouteCategory.CASUAL, None),
}

STATIC_CATEGORY_RESPONSES: Dict[RouteCategory, str] = {
    RouteCategory.CASUAL: (
        "Hello! I am your Enterprise HR Intelligence Assistant. "
        "I can help with company HR policies, benefits, leave, expenses, and employee information. "
        "How can I help you today?"
    ),
    RouteCategory.OUT_OF_SCOPE: (
        "I'm designed to help with company HR policies, employee information, "
        "benefits, leave, expenses, and related workplace questions. "
        "Your query falls outside of approved enterprise documentation."
    ),
}

DEFAULT_FALLBACK_ANSWER = (
    "I could not determine the specific enterprise department for this question. "
    "Please ask about company HR policies, benefits, leave, or employee counts."
)
