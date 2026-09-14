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
    HF_TOKEN: SecretStr | None = None

    # Default LLM configuration
    DEFAULT_PROVIDER: Literal["gemini", "openai", "groq", "ollama"] = "ollama"
    DEFAULT_MODEL: str = "qwen2.5-coder:1.5b"
    OPENAI_MODEL: str = "gpt-4o-mini"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    OLLAMA_MODEL: str = "qwen2.5-coder:1.5b"
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
    LANGSMITH_PROJECT: str = "people-query-hr"

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
    MAX_CONVERSATION_HISTORY: int = Field(
        default=10,
        gt=0,
        le=100,
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

    # LLM-as-a-Judge Configuration
    ENABLE_LLM_JUDGE: bool = True
    JUDGE_PROVIDER: Literal["gemini", "openai", "groq", "ollama"] | None = None
    JUDGE_MODEL: str | None = None
    JUDGE_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    MAX_JUDGE_RETRIES: int = Field(
        default=1,
        ge=0,
        le=5,
    )
    JUDGE_TIMEOUT_SECONDS: int = Field(
        default=10,
        gt=0,
        le=60,
    )
    JUDGE_SCORE_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    # Evaluation & Loop Bounds
    MAX_EVALUATION_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )
    MAX_SQL_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )


    # Guardrails
    ALLOWED_SQL_TABLES: list[str] = Field(
        default_factory=lambda: [
            "departments",
            "positions",
            "employees",
            "leaves",
            "benefits",
            "employee_benefits",
            "performance_reviews",
        ],
    )
    BLOCKED_SQL_KEYWORDS: list[str] = Field(
        default_factory=lambda: [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "REPLACE",
            "GRANT",
            "REVOKE",
            "EXEC",
            "EXECUTE",
            "ATTACH",
            "DETACH",
            "PRAGMA",
        ],
    )
    MAX_INPUT_LENGTH: int = Field(default=2000, gt=0, le=10_000)
    REQUIRE_CITATIONS_FOR_RAG: bool = True
    MASK_INDIVIDUAL_SALARIES: bool = True


    # Third-party environment configuration

    def configure_environment(self) -> None:
        """
        Export settings to os.environ for third-party libraries
        such as LangSmith/LangChain.
        """
        if not self.LANGSMITH_TRACING or not self.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            return

        api_key = self.LANGSMITH_API_KEY.get_secret_value()
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT

        # Also set LangChain standard environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT

        # Export Hugging Face token for faster downloads & rate limit authorization
        if self.HF_TOKEN:
            hf_val = self.HF_TOKEN.get_secret_value()
            os.environ["HF_TOKEN"] = hf_val
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_val
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_val


@lru_cache
def get_settings() -> Settings:
    """
    Cached application settings.

    Settings are created once and reused throughout the application.
    """

    settings = Settings()
    settings.configure_environment()

    return settings