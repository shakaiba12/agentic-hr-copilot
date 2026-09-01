"""
Tests for the SQL Pipeline: SchemaProvider and SQLExecutor.
Generator tests are skipped without a live API key (tested via integration).
"""
import pytest
from src.core.config import get_settings
from src.sql.schema_provider import SchemaProvider
from src.sql.executor import SQLExecutor, ExecutionResult
from src.guardrails.sql_guardrail import SQLGuardrail


class TestSchemaProvider:
    """Unit tests for SchemaProvider."""

    def setup_method(self):
        self.provider = SchemaProvider()

    def test_full_schema_includes_all_tables(self):
        schema = self.provider.get_full_schema()
        for table in ["employees", "departments", "positions", "leaves", "benefits"]:
            assert table in schema, f"Expected table '{table}' in schema"

    def test_full_schema_includes_column_names(self):
        schema = self.provider.get_full_schema()
        assert "hire_date" in schema
        assert "department_id" in schema
        assert "employment_status" in schema

    def test_full_schema_includes_fk_section(self):
        schema = self.provider.get_full_schema()
        assert "Foreign key" in schema or "->" in schema

    def test_relevant_schema_filters_to_requested_tables(self):
        schema = self.provider.get_relevant_schema(["employees", "departments"])
        assert "employees" in schema
        assert "departments" in schema
        assert "performance_reviews" not in schema

    def test_relevant_schema_falls_back_to_full_when_no_hints(self):
        schema_full = self.provider.get_full_schema()
        schema_fallback = self.provider.get_relevant_schema(hint_tables=None)
        assert schema_full == schema_fallback

    def test_relevant_schema_handles_unknown_table_gracefully(self):
        schema = self.provider.get_relevant_schema(["nonexistent_table"])
        # Should return something (with header) but no table body
        assert isinstance(schema, str)


class TestSQLExecutor:
    """Unit tests for SQLExecutor against the seeded SQLite database."""

    def setup_method(self):
        self.executor = SQLExecutor()

    def test_returns_all_departments(self):
        result = self.executor.execute("SELECT id, name FROM departments ORDER BY id")
        assert result.success is True
        assert result.row_count == 6
        names = [r["name"] for r in result.rows]
        assert "Engineering" in names
        assert "Finance" in names

    def test_counts_employees_correctly(self):
        result = self.executor.execute("SELECT COUNT(*) AS total FROM employees")
        assert result.success is True
        assert result.rows[0]["total"] == 25

    def test_join_employees_with_departments(self):
        sql = (
            "SELECT e.first_name, d.name AS dept "
            "FROM employees e JOIN departments d ON e.department_id = d.id "
            "WHERE d.name = 'Engineering'"
        )
        result = self.executor.execute(sql)
        assert result.success is True
        assert result.row_count > 0
        assert all(r["dept"] == "Engineering" for r in result.rows)

    def test_tenure_filter_returns_eligible_employees(self):
        """Employees hired more than 12 months ago — maternity leave eligible."""
        sql = (
            "SELECT COUNT(*) AS eligible "
            "FROM employees "
            "WHERE hire_date <= date('now', '-12 months') "
            "AND employment_type = 'FULL_TIME'"
        )
        result = self.executor.execute(sql)
        assert result.success is True
        assert result.rows[0]["eligible"] >= 10  # seeded data has many tenured employees

    def test_empty_result_is_handled(self):
        result = self.executor.execute(
            "SELECT id FROM employees WHERE first_name = 'NOBODY_EXISTS_XYZ'"
        )
        assert result.success is True
        assert result.row_count == 0
        assert result.rows == []

    def test_invalid_sql_returns_failure(self):
        result = self.executor.execute("SELECT FROM WHERE nobody")
        assert result.success is False
        assert result.error != ""

    def test_row_truncation_flag(self):
        """Executor with max_rows=1 should set was_truncated=True for multi-row results."""
        settings = get_settings()
        # Directly wire a minimal executor with max_rows overridden
        executor = SQLExecutor.__new__(SQLExecutor)
        executor._db_path = self.executor._db_path
        executor._max_rows = 1
        executor._timeout = settings.DB_QUERY_TIMEOUT_SECONDS

        result = executor.execute("SELECT id FROM employees ORDER BY id")
        assert result.was_truncated is True
        assert result.row_count == 1


class TestSQLGuardrailIntegration:
    """Integration: SQLGuardrail + SQLExecutor form a safe pipeline."""

    def setup_method(self):
        self.guardrail = SQLGuardrail()
        self.executor = SQLExecutor()

    def test_safe_query_passes_and_executes(self):
        sql = "SELECT COUNT(*) AS n FROM employees"
        guard_result = self.guardrail.validate(sql)
        assert guard_result.is_valid is True

        exec_result = self.executor.execute(guard_result.normalized_sql)
        assert exec_result.success is True
        assert exec_result.rows[0]["n"] == 25

    def test_destructive_query_is_blocked_before_execution(self):
        sql = "DROP TABLE employees"
        guard_result = self.guardrail.validate(sql)
        assert guard_result.is_valid is False
        # Never reached executor — no side effects
