"""
PeopleQuery AI - Agentic HR Analytics Copilot
Interactive CLI Entrypoint with Master Orchestration Gate
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
from src.core.orchestrator import MasterOrchestrator, OrchestratorResponse


def print_banner(settings):
    print("=" * 65)
    print(" PeopleQuery AI — Enterprise HR Intelligence Copilot")
    print("=" * 65)
    print(f" Environment : {settings.APP_ENV}")
    print(f" LLM Provider: {settings.DEFAULT_PROVIDER} ({settings.DEFAULT_MODEL})")
    print(f" Database    : {settings.DATABASE_URL}")
    print(f" LangSmith   : {'Enabled' if settings.LANGSMITH_TRACING else 'Disabled'}")
    print("=" * 65)
    print("Type your HR question below, or 'exit' / 'quit' to close.\n")


def _clean_user_input(raw_input: str) -> str | None:
    """Clean pasted prompt markers or discard accidental terminal log lines."""
    line = raw_input.strip()
    if not line:
        return None

    # Strip pasted prompt markers
    for prefix in ("User ❯ ", "User > ", "User: ", "User ❯", "User >", "User:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()

    # Discard pure terminal output fragments / log markers
    _IGNORED_PREFIXES = (
        "[Orchestrator]",
        "Category  :",
        "Target    :",
        "Allowed   :",
        "Confidence:",
        "Reason    :",
        "💬 [Master Response]:",
        "🛑 [Master Blocked]:",
        "📚 [RAG Pipeline Response]:",
        "📜 [Generated SQL]:",
        "✅ [Database Result]",
        "❌ [SQL Pipeline Error]:",
        "RAG Knowledge Handler received query:",
        "SELECT ",
        "FROM ",
        "JOIN ",
        "WHERE ",
        "LIMIT ",
        "[Database Result]",
        "SQL Pipeline Error",
        "Master Blocked",
        "Master Response",
        "ted SQL]:",
    )
    if any(line.startswith(p) for p in _IGNORED_PREFIXES):
        return None

    return line if line else None


def main():
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_banner(settings)

    # Initialize Master Orchestrator (Router + Guardrails + RAG + SQL Pipelines)
    orchestrator = MasterOrchestrator(settings=settings)
    history: list[dict] = []

    while True:
        try:
            raw_input = input("User ❯ ")
            clean_input = _clean_user_input(raw_input)

            if clean_input is None:
                if not raw_input.strip():
                    print("\n ℹ️  [Notice]: No query provided. Please enter an HR question.\n")
                continue

            if clean_input.lower() in ("exit", "quit", "q"):
                print("\n Goodbye!")
                break

            # Explicit Logging Boundary
            print(f"\n [USER INPUT] \"{clean_input}\"")

            # Execute Master Orchestrator Gate
            result: OrchestratorResponse = orchestrator.process_query(clean_input, history=history)

            # Observable Routing Display
            decision = result.decision
            print(f"\n [Orchestrator]")
            print(f" Category  : {decision.category.value}")
            print(f" Target    : {decision.target or 'None (Blocked)'}")
            print(f" Allowed   : {decision.allowed}")
            print(f" Confidence: {decision.confidence:.1f}")
            print(f" Reason    : {decision.reason}")

            # Display Response Based on Handler Source
            if result.source == "master":
                if not result.allowed:
                    if decision.category.value == "INVALID":
                        print(f"\n {result.response}\n")
                    else:
                        print(f"\n 🛑 [Master Blocked]: {result.response}\n")
                else:
                    print(f"\n 💬 [Master Response]: {result.response}\n")

            elif result.source == "rag":
                print(f"\n 📚 [RAG Pipeline Response]:")
                print(f"    {result.response}\n")

            elif result.source == "sql":
                sql_res = result.sql_result
                if sql_res and sql_res.success:
                    print(f"\n 📜 [Generated SQL]: {sql_res.generated_sql}")
                    print(f" ✅ [Database Result] ({sql_res.row_count} rows returned):")
                    for row in sql_res.rows:
                        print(f"    {row}")
                    if sql_res.execution and sql_res.execution.was_truncated:
                        print("    ... (results truncated to maximum row cap)")
                    print()
                else:
                    print(f"\n ❌ [SQL Pipeline Error]: {result.response}\n")

            # Maintain strictly role-separated conversation history for follow-ups
            history.append({
                "role": "user",
                "content": clean_input,
                "category": decision.category,
                "decision": decision,
            })
            history.append({
                "role": "assistant",
                "content": result.response,
                "source": result.source,
            })

        except (KeyboardInterrupt, EOFError):
            print("\n Session ended.")
            break


if __name__ == "__main__":
    main()

