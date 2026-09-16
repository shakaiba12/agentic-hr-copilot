"""
PeopleQuery AI - API Server Entrypoint
Run with: python server.py
"""

import uvicorn
from src.core.config import get_settings


def main():
    settings = get_settings()
    print("=" * 65)
    print(" Starting PeopleQuery AI Backend API Server")
    print(f" Environment : {settings.APP_ENV}")
    print(f" LLM Provider: {settings.DEFAULT_PROVIDER} ({settings.DEFAULT_MODEL})")
    print(f" Database    : {settings.DATABASE_URL}")
    print(f" API URL     : http://127.0.0.1:8000")
    print(f" OpenAPI Docs: http://127.0.0.1:8000/docs")
    print("=" * 65)

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
