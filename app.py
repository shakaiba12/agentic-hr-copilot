"""
PeopleQuery AI - Agentic HR Analytics Copilot
Interactive CLI Entrypoint
"""
import sys
import logging
import warnings

# Suppress verbose third-party logger outputs in CLI
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from src.core.config import get_settings
from src.core.state import AgentState, IntentType
from src.guardrails import InputGuardrail, SQLGuardrail
from src.sql import SchemaProvider, SQLGenerator, SQLExecutor



def print_banner(settings):
    print("=" * 65)
    print(" PeopleQuery AI — Single AI HR Intelligence Copilot")
    print("=" * 65)
    print(f" Environment : {settings.APP_ENV}")
    print(f" LLM Provider: {settings.DEFAULT_PROVIDER} ({settings.DEFAULT_MODEL})")
    print(f" Database    : {settings.DATABASE_URL}")
    print(f" LangSmith   : {'Enabled' if settings.LANGSMITH_TRACING else 'Disabled'}")
    print("=" * 65)
    print("Type your HR question below, or 'exit' / 'quit' to close.\n")


def main():
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_banner(settings)

    # Initialize components
    input_guard = InputGuardrail(settings)
    sql_guard = SQLGuardrail(settings)
    schema_provider = SchemaProvider()
    sql_executor = SQLExecutor()

    while True:
        try:
            user_input = input("User ❯ ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\n Goodbye!")
                break

            # 1. Input Guardrail
            input_check = input_guard.check(user_input)
            if not input_check.is_safe:
                print(f"\n 🛑 [Input Guardrail Blocked]: {input_check.rejection_reason}\n")
                continue

            # 2. Schema Introspection
            schema_context = schema_provider.get_full_schema()

            # 3. SQL Generation via LLM
            print("\n 🔍 [SQL Pipeline] Generating SQL query...")
            try:
                sql_gen = SQLGenerator()
                gen_result = sql_gen.generate(input_check.sanitized_query, schema_context)
            except Exception as exc:
                print(f" ⚠️  [LLM Error]: Could not generate SQL ({exc})\n")
                continue

            if not gen_result.is_generatable:
                print(f" ⚠️  [SQL Generator]: {gen_result.reason}\n")
                continue

            generated_sql = gen_result.sql
            print(f" 📜 [Generated SQL]: {generated_sql}")

            # 4. SQL Safety Guardrail
            sql_check = sql_guard.validate(generated_sql)
            if not sql_check.is_valid:
                print(f" 🛑 [SQL Guardrail Blocked]: {sql_check.notes}\n")
                continue

            # 5. Database Execution
            exec_result = sql_executor.execute(sql_check.normalized_sql)
            if not exec_result.success:
                print(f" ❌ [Database Error]: {exec_result.error}\n")
                continue

            # Display Execution Results
            print(f" ✅ [Database Result] ({exec_result.row_count} rows returned):")
            for row in exec_result.rows:
                print(f"    {row}")
            if exec_result.was_truncated:
                print("    ... (results truncated to maximum row cap)")
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n Session ended.")
            break


if __name__ == "__main__":
    main()
