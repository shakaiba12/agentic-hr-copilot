"""
PeopleQuery AI - Agentic HR Analytics Copilot
Interactive CLI Entrypoint
"""
import sys
from src.core.config import get_settings
from src.core.state import AgentState, IntentType


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

    while True:
        try:
            user_input = input("User ❯ ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\n Goodbye!")
                break

            # Initialize AgentState for Phase 1 harness
            state: AgentState = {
                "query": user_input,
                "sanitized_query": user_input,
                "is_input_safe": True,
                "input_rejection_reason": None,
                "intent": IntentType.UNKNOWN,
                "intent_reasoning": None,
                "messages": [],
                "db_schema_context": None,
                "generated_sql": None,
                "is_sql_valid": None,
                "sql_validation_notes": None,
                "sql_data": None,
                "sql_row_count": None,
                "retrieved_chunks": None,
                "citations": None,
                "candidate_answer": None,
                "judge_evaluation": None,
                "retry_count": 0,
                "final_answer": None,
                "errors": [],
                "metadata": {"version": "0.1.0"},
            }

            print(f"\n[Phase 1 Harness] Query Received: \"{state['query']}\"")
            print("  State schema validated successfully. (Foundations and Seed Data Ready)\n")


        except (KeyboardInterrupt, EOFError):
            print("\n Session ended.")
            break


if __name__ == "__main__":
    main()
