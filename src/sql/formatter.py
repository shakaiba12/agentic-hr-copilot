"""
SQL Result Presentation and Formatting Module.
Uses LLM-driven presentation decisions with lightweight deterministic rendering safeguards.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.core.llm import get_llm

logger = logging.getLogger(__name__)


class SQLPresentation(BaseModel):
    """Structured presentation decision from the LLM."""

    response_type: Literal["answer", "table", "answer_and_table"] = Field(
        ...,
        description="Presentation type: 'answer' for scalar/entity answers, 'table' for explicit records/grouped data, 'answer_and_table' for summary plus records.",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Concise natural language answer. Required when response_type is 'answer' or 'answer_and_table'.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="List of relevant column keys from the SQL result to include in the table.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief summary phrase preceding the table when response_type is 'answer_and_table'.",
    )


PRESENTATION_SYSTEM_PROMPT = """You are the SQL Result Presentation Assistant for an Enterprise HR Copilot.
Your job is to examine a user's question, the executed SQL query, and the database query results, and decide on the smallest, most useful, user-friendly presentation format.

PRESENTATION MODES:
1. "answer" (NO TABLE):
   - Use when the user asks for a count, scalar aggregate (average, highest, lowest, sum, total, rating), or single entity lookup.
   - Even if the database returned multiple rows, if the user's question is asking for an aggregate/count (e.g. "How many employees are there?"), return "answer".
   - Ground the natural language answer strictly in the SQL results.
   - Format currency ($182,500), dates, percentages, and names naturally in the answer.
   - Examples:
     * "How many employees are there?" -> response_type: "answer", answer: "10 employees."
     * "How many active employees are there?" -> response_type: "answer", answer: "10 active employees."
     * "What is the average salary?" -> response_type: "answer", answer: "The average salary is $182,500."
     * "What is the highest salary?" -> response_type: "answer", answer: "The highest salary is $260,000."
     * "Who has the highest salary?" / "Who is the highest paid employee?" -> response_type: "answer", answer: "Elena Rostova has the highest salary at $260,000."
     * "What is the average performance rating?" -> response_type: "answer", answer: "The average performance rating is 4.45."
     * "Tell me about Elena Rostova." -> response_type: "answer", answer: "Elena Rostova is a VP of Engineering in Engineering with a salary of $260,000 (Status: Active)."

2. "table" (EXACTLY ONE TABLE):
   - Use when the user explicitly asks to list, show, or browse multiple records (e.g. "Show me all employees", "List employees earning more than $200k", "Show employees in Engineering").
   - Use when the user asks for grouped aggregate results (e.g. "How many employees are in each department?").
   - Specify the list of relevant columns in `columns`. Do NOT include unrequested internal database IDs (like id, employee_id, department_id, position_id).

3. "answer_and_table":
   - Use only when the user explicitly requests both a summary and the underlying records.

CRITICAL RULES:
- Never hallucinate numbers, names, or facts. The SQL result is the sole source of truth.
- Output MUST be a valid JSON object matching the schema:
{
  "response_type": "answer" | "table" | "answer_and_table",
  "answer": "concise answer string or null",
  "columns": ["col1", "col2"] or null,
  "summary": "short summary or null"
}
"""


def _format_currency(val: Any) -> str:
    """Safely format a numeric value as currency."""
    try:
        num = float(val)
        if num.is_integer():
            return f"${int(num):,}"
        return f"${num:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def _format_rating_or_float(val: Any) -> str:
    """Format floating point numbers like ratings."""
    try:
        num = float(val)
        if num.is_integer():
            return str(int(num))
        formatted = f"{num:.2f}"
        if formatted.endswith("0"):
            return formatted.rstrip("0")
        return formatted
    except (ValueError, TypeError):
        return str(val)


def _is_currency_column(col_name: str) -> bool:
    """Check if column name represents currency / compensation."""
    col = col_name.lower()
    return any(k in col for k in ("salary", "budget", "compensation", "bonus", "pay", "min_salary", "max_salary"))


def _is_rating_column(col_name: str) -> bool:
    """Check if column name represents rating / score."""
    col = col_name.lower()
    return any(k in col for k in ("rating", "score", "performance"))


def _is_unnecessary_column(col_name: str, question: Optional[str] = None) -> bool:
    """Check if column is an internal ID or unrequested field (like email, gender, etc.)."""
    col = col_name.lower()
    q = (question or "").lower()

    if col in ("id", "employee_id", "manager_id", "benefit_id", "doc_id"):
        return "id" not in q

    if col == "email" and not any(k in q for k in ("email", "contact", "address", "reach")):
        return True

    if col == "gender" and not any(k in q for k in ("gender", "sex", "demographic")):
        return True

    if col == "employment_type" and not any(k in q for k in ("type", "full time", "part time", "contract")):
        return True

    return False


def render_markdown_table(
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    question: Optional[str] = None,
    max_rows: int = 25,
) -> str:
    """
    Safely render a list of dictionary rows into exactly ONE formatted Markdown table.
    - Selects relevant columns (filtering internal IDs and unnecessary fields unless requested).
    - Combines first_name and last_name into Name when present.
    - Formats currency, dates, and ratings.
    - Uses human-readable column headers.
    """
    if not rows:
        return ""

    sample_row = rows[0]
    available_cols = list(sample_row.keys())

    # Determine columns to display
    if columns:
        selected_cols = [
            c for c in columns
            if c in available_cols and not _is_unnecessary_column(c, question)
        ]
        if not selected_cols:
            selected_cols = [c for c in columns if c in available_cols]
        if not selected_cols:
            selected_cols = available_cols
    else:
        # Default: omit internal IDs & unrequested sensitive fields
        selected_cols = [c for c in available_cols if not _is_unnecessary_column(c, question)]
        if not selected_cols:
            selected_cols = available_cols

    # Check if first_name and last_name can be combined into Name
    combine_name = "first_name" in selected_cols and "last_name" in selected_cols

    display_headers: List[str] = []
    col_mapping: List[Tuple[str, Optional[str]]] = []  # (col_key or '__name__', extra_col)

    if combine_name:
        display_headers.append("Name")
        col_mapping.append(("__name__", None))
        for col in selected_cols:
            if col not in ("first_name", "last_name"):
                display_headers.append(_format_header(col))
                col_mapping.append((col, None))
    else:
        for col in selected_cols:
            display_headers.append(_format_header(col))
            col_mapping.append((col, None))

    # Format table rows
    header_line = "| " + " | ".join(display_headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in display_headers) + " |"
    data_lines = []

    for row in rows[:max_rows]:
        line_cells: List[str] = []
        for map_key, _ in col_mapping:
            if map_key == "__name__":
                fn = str(row.get("first_name", "")).strip()
                ln = str(row.get("last_name", "")).strip()
                full_name = f"{fn} {ln}".strip() or "—"
                line_cells.append(full_name)
            else:
                val = row.get(map_key)
                line_cells.append(_format_cell_value(map_key, val))
        data_lines.append("| " + " | ".join(line_cells) + " |")

    table_md = "\n".join([header_line, sep_line] + data_lines)
    if len(rows) > max_rows:
        table_md += f"\n\n*(Showing top {max_rows} of {len(rows)} records)*"

    return table_md


def _format_header(col: str) -> str:
    """Convert column key to human-readable header title."""
    header_map = {
        "department_id": "Department",
        "position_id": "Position",
        "employment_status": "Status",
        "employment_type": "Employment Type",
        "hire_date": "Hire Date",
        "performance_rating": "Performance Rating",
        "employee_count": "Employee Count",
        "count(*)": "Count",
        "count": "Count",
    }
    col_lower = col.lower()
    if col_lower in header_map:
        return header_map[col_lower]
    return col.replace("_", " ").title()


def _format_cell_value(col_key: str, val: Any) -> str:
    """Format individual cell value according to data type and column semantics."""
    if val is None or val == "":
        return "—"
    if _is_currency_column(col_key):
        return _format_currency(val)
    if _is_rating_column(col_key):
        return _format_rating_or_float(val)
    if isinstance(val, float):
        return f"{val:.2f}".rstrip("0").rstrip(".") if f"{val:.2f}".endswith(".00") else f"{val:.2f}"
    return str(val)


class SQLResultPresenter:
    """
    LLM-driven SQL Result Presenter.
    Determines whether a concise natural-language answer, a single formatted table,
    or a summary plus table should be presented to the user.
    """

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature

    def present(
        self,
        question: str,
        rows: List[Dict[str, Any]],
        row_count: int,
        generated_sql: Optional[str] = None,
        is_valid: bool = True,
        sql_error: Optional[str] = None,
    ) -> str:
        """
        Produce the final user-facing SQL answer string.
        """
        if not is_valid:
            return sql_error or "The SQL query could not be validated."

        if sql_error:
            return f"Database execution error: {sql_error}"

        if not rows or row_count == 0:
            return "No matching employee or workforce records were found in the database."

        # Call LLM to decide presentation format
        presentation = self._get_llm_presentation(
            question=question,
            rows=rows,
            row_count=row_count,
            generated_sql=generated_sql,
        )

        # Render output safely based on LLM decision
        return self._render_presentation(presentation, rows, row_count, question)

    def _get_llm_presentation(
        self,
        question: str,
        rows: List[Dict[str, Any]],
        row_count: int,
        generated_sql: Optional[str],
    ) -> SQLPresentation:
        """Query LLM for structured presentation decision."""
        try:
            llm = get_llm(temperature=self.temperature)

            # Sample rows for context (up to 10 rows to keep prompt fast and light)
            sample_data = rows[:10]
            columns = list(rows[0].keys()) if rows else []

            user_prompt = (
                f"USER QUESTION: \"{question}\"\n"
                f"GENERATED SQL: {generated_sql or 'N/A'}\n"
                f"RESULT COLUMNS: {columns}\n"
                f"TOTAL ROW COUNT: {row_count}\n"
                f"SQL EXECUTION RESULT (sample up to 10 rows):\n{json.dumps(sample_data, default=str)}\n\n"
                "Return your presentation JSON object now:"
            )

            messages = [
                SystemMessage(content=PRESENTATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)

            return self._parse_presentation_json(raw_content)

        except Exception as exc:
            logger.warning("LLM presentation decision failed, using deterministic fallback: %s", exc)
            return self._deterministic_fallback(question, rows, row_count)

    def _parse_presentation_json(self, raw_content: str) -> SQLPresentation:
        """Safely parse JSON response into SQLPresentation model."""
        text = raw_content.strip()
        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        # Extract JSON object matching { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        data = json.loads(text)
        resp_type = data.get("response_type", "answer")
        if resp_type not in ("answer", "table", "answer_and_table"):
            resp_type = "answer"

        return SQLPresentation(
            response_type=resp_type,
            answer=data.get("answer"),
            columns=data.get("columns"),
            summary=data.get("summary"),
        )

    def _render_presentation(
        self,
        pres: SQLPresentation,
        rows: List[Dict[str, Any]],
        row_count: int,
        question: str,
    ) -> str:
        """Render the final user-facing text from the structured presentation."""
        if pres.response_type == "answer":
            if pres.answer and pres.answer.strip():
                return pres.answer.strip()
            # Fallback if answer string was unexpectedly blank
            return self._deterministic_answer_fallback(rows, question)

        if pres.response_type == "table":
            table_md = render_markdown_table(rows, columns=pres.columns, question=question)
            return table_md if table_md else self._deterministic_answer_fallback(rows, question)

        if pres.response_type == "answer_and_table":
            table_md = render_markdown_table(rows, columns=pres.columns, question=question)
            summary = pres.summary or pres.answer or f"Found {row_count} records:"
            if table_md:
                return f"{summary.strip()}\n\n{table_md}"
            return summary.strip()

        # Default fallback
        return render_markdown_table(rows, columns=pres.columns)

    def _deterministic_fallback(
        self,
        question: str,
        rows: List[Dict[str, Any]],
        row_count: int,
    ) -> SQLPresentation:
        """Lightweight fallback when LLM is unavailable."""
        q_lower = question.lower()

        # 1. Scalar / single value aggregate
        if len(rows) == 1 and len(rows[0]) == 1:
            val = next(iter(rows[0].values()))
            if "how many" in q_lower or "count" in q_lower:
                entity = "active employees" if "active" in q_lower else "employees"
                ans = f"{val} {entity}."
            elif "average salary" in q_lower or "avg salary" in q_lower:
                ans = f"The average salary is {_format_currency(val)}."
            elif "highest salary" in q_lower or "maximum salary" in q_lower:
                ans = f"The highest salary is {_format_currency(val)}."
            elif "lowest salary" in q_lower or "minimum salary" in q_lower:
                ans = f"The lowest salary is {_format_currency(val)}."
            elif "rating" in q_lower or "performance" in q_lower:
                ans = f"The average performance rating is {_format_rating_or_float(val)}."
            else:
                ans = f"{val}."
            return SQLPresentation(response_type="answer", answer=ans)

        # 2. Single record lookup
        if len(rows) == 1:
            row = rows[0]
            name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
            if "highest" in q_lower and "salary" in q_lower and "salary" in row:
                ans = f"{name or 'The employee'} has the highest salary at {_format_currency(row['salary'])}."
                return SQLPresentation(response_type="answer", answer=ans)
            if "lowest" in q_lower and "salary" in q_lower and "salary" in row:
                ans = f"{name or 'The employee'} has the lowest salary at {_format_currency(row['salary'])}."
                return SQLPresentation(response_type="answer", answer=ans)

        # 3. Explicit multi-row or grouped request -> table
        return SQLPresentation(response_type="table", columns=list(rows[0].keys()) if rows else None)

    def _deterministic_answer_fallback(self, rows: List[Dict[str, Any]], question: str) -> str:
        """Safely extract scalar or concise string when answer text is missing."""
        if len(rows) == 1 and len(rows[0]) == 1:
            val = next(iter(rows[0].values()))
            return f"{val}."
        if len(rows) == 1:
            row = rows[0]
            if "first_name" in row and "last_name" in row and "salary" in row:
                return f"{row['first_name']} {row['last_name']} ({_format_currency(row['salary'])})."
        return render_markdown_table(rows)
