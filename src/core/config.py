import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata

    PROJECT_NAME: str = "PeopleQuery AI"
    APP_ENV: str = "development"
    DEBUG: bool = False

    # LLM API Keys

    GEMINI_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None

    # Default LLM configuration
    DEFAULT_PROVIDER: Literal["gemini", "openai", "groq", "ollama"] = "gemini"
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OLLAMA_MODEL: str = "deepseek-r1:latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    # LangSmith Observability
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str = "peoplequery-ai"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./data/hr_database.sqlite"
    DB_QUERY_TIMEOUT_SECONDS: int = Field(
        default=15,
        gt=0,
    )
    DB_MAX_ROWS_RETURNED: int = Field(
        default=100,
        gt=0,
        le=10_000,
    )

    # RAG & Chroma Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DOCS_DIR: Path = BASE_DIR / "company_docs"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    TOP_K_RETRIEVAL: int = Field(
        default=4,
        gt=0,
        le=50,
    )
    BM25_TOP_K: int = Field(
        default=4,
        gt=0,
        le=50,
    )

    # Evaluation & Loop Bounds
    MAX_EVALUATION_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )


    # Third-party environment configuration

    def configure_environment(self) -> None:
        """
        Export settings to os.environ for third-party libraries
        such as LangSmith/LangChain.
        """

        if not self.LANGSMITH_TRACING:
            return

        if not self.LANGSMITH_API_KEY:
            return

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_API_KEY"] = (
            self.LANGSMITH_API_KEY.get_secret_value()
        )
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT


@lru_cache
def get_settings() -> Settings:
    """
    Cached application settings.

    Settings are created once and reused throughout the application.
    """

    settings = Settings()
    settings.configure_environment()

    return settings