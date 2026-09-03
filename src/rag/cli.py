"""
Interactive CLI entrypoint for manual chunk inspection and approval (Step 1).
Avoids module import cycles when running with `python -m src.rag.cli`.
"""

from src.rag.chunking import run_cli_interactive

if __name__ == "__main__":
    run_cli_interactive()
