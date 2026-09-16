"""
Curated SQL Exemplars for HR Domain Queries.
Provides a minimal, static repository of high-quality (Question -> SQL) pairs
to guide LLM generation on tricky SQL patterns without vector retrieval overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class SQLExemplar:
    question: str
    sql: str
    tags: tuple[str, ...]


_EXEMPLARS: tuple[SQLExemplar, ...] = (
    SQLExemplar(
        question="What is the average metric and count grouped by category?",
        sql=(
            "SELECT c.name AS category_name, "
            "COUNT(t.id) AS total_count, "
            "ROUND(AVG(t.metric_value), 2) AS average_value "
            "FROM categories c "
            "LEFT JOIN table_items t ON c.id = t.category_id "
            "GROUP BY c.name "
            "ORDER BY average_value DESC;"
        ),
        tags=("average", "count", "headcount", "group by", "summary", "stats", "total", "by"),
    ),
    SQLExemplar(
        question="Which records were created in the last 12 months?",
        sql=(
            "SELECT id, name, status, event_date "
            "FROM table_items "
            "WHERE event_date >= date('now', '-12 months') "
            "ORDER BY event_date DESC "
            "LIMIT 100;"
        ),
        tags=("date", "recent", "months", "days", "years", "time", "since", "after", "before"),
    ),
    SQLExemplar(
        question="Find matching records with partial text filtering and multi-table joins",
        sql=(
            "SELECT a.name, b.title, a.status "
            "FROM table_a a "
            "JOIN table_b b ON a.relation_id = b.id "
            "WHERE LOWER(b.title) LIKE '%keyword%' "
            "LIMIT 100;"
        ),
        tags=("join", "filter", "like", "find", "search", "lookup", "where", "matching"),
    ),
)


def get_relevant_exemplars(query: str, max_exemplars: int = 2) -> List[SQLExemplar]:
    """
    Select top relevant SQL pattern exemplars based on keyword/tag overlap.
    """
    if not query:
        return list(_EXEMPLARS[:max_exemplars])

    q_lower = query.lower()
    scores: list[tuple[int, SQLExemplar]] = []

    for exemplar in _EXEMPLARS:
        score = 0
        for tag in exemplar.tags:
            if tag in q_lower:
                score += 2
        for word in exemplar.question.lower().split():
            if len(word) > 3 and word in q_lower:
                score += 1
        scores.append((score, exemplar))

    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [ex for score, ex in scores if score > 0][:max_exemplars]

    if not selected:
        selected = list(_EXEMPLARS[:max_exemplars])

    return selected


def format_exemplars_for_prompt(exemplars: List[SQLExemplar]) -> str:
    """Format exemplar list as LLM few-shot context."""
    if not exemplars:
        return ""

    lines = ["\nExamples of valid SQL queries for similar questions:"]
    for i, ex in enumerate(exemplars, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Question: \"{ex.question}\"")
        lines.append(f"SQL:\n{ex.sql}\n")

    return "\n".join(lines)
