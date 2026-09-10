import logging
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_fallbacks: bool = True,
    **kwargs,
) -> BaseChatModel:
    """
    Create and return a LangChain chat model with automatic rate-limit & provider fallbacks.
    """
    settings = get_settings()

    provider_name = (provider or settings.DEFAULT_PROVIDER).lower()
    temp = (
        temperature
        if temperature is not None
        else settings.DEFAULT_TEMPERATURE
    )

    primary_llm = _create_single_llm(
        provider=provider_name,
        model_name=model_name,
        temperature=temp,
        settings=settings,
        **kwargs,
    )

    if not enable_fallbacks:
        return primary_llm

    # Build fallback models to handle rate limits (429) or provider outages
    fallbacks: List[BaseChatModel] = []

    # 1. Intra-provider fallbacks for Groq
    if provider_name == "groq":
        current_model = model_name or settings.GROQ_MODEL
        groq_alternatives = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for alt_model in groq_alternatives:
            if alt_model != current_model:
                try:
                    alt_llm = _create_groq(
                        settings=settings,
                        model_name=alt_model,
                        temperature=temp,
                        **kwargs,
                    )
                    fallbacks.append(alt_llm)
                except Exception:
                    pass

    # 2. Cross-provider fallback to Gemini if configured
    if provider_name != "gemini" and _get_secret(settings.GEMINI_API_KEY):
        try:
            gemini_llm = _create_gemini(
                settings=settings,
                model_name=None,
                temperature=temp,
                **kwargs,
            )
            fallbacks.append(gemini_llm)
        except Exception:
            pass

    # 3. Cross-provider fallback to OpenAI if configured
    if provider_name != "openai" and _get_secret(settings.OPENAI_API_KEY):
        try:
            openai_llm = _create_openai(
                settings=settings,
                model_name=settings.OPENAI_MODEL,
                temperature=temp,
                **kwargs,
            )
            fallbacks.append(openai_llm)
        except Exception:
            pass

    # 4. Fallback to local Ollama if available
    if provider_name != "ollama" and settings.OLLAMA_BASE_URL:
        try:
            ollama_llm = _create_ollama(
                settings=settings,
                model_name=settings.OLLAMA_MODEL or "qwen2.5-coder:1.5b",
                temperature=temp,
                **kwargs,
            )
            fallbacks.append(ollama_llm)
        except Exception:
            pass

    if fallbacks:
        try:
            return primary_llm.with_fallbacks(fallbacks)
        except Exception as e:
            logger.debug(f"Could not attach LLM fallbacks: {e}")

    return primary_llm


def _create_single_llm(
    provider: str,
    model_name: Optional[str],
    temperature: float,
    settings,
    **kwargs,
) -> BaseChatModel:
    """Create a single chat model instance without fallbacks."""
    if provider == "gemini":
        return _create_gemini(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    if provider == "openai":
        return _create_openai(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    if provider == "groq":
        return _create_groq(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    if provider == "ollama":
        return _create_ollama(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported providers: gemini, openai, groq, ollama."
    )



def _get_secret(key_obj) -> Optional[str]:
    if key_obj is None:
        return None
    val = key_obj.get_secret_value() if hasattr(key_obj, "get_secret_value") else str(key_obj)
    return val if val.strip() else None


def _create_gemini(
    settings,
    model_name: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    api_key = _get_secret(settings.GEMINI_API_KEY)
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "Install Gemini support with: "
            "pip install langchain-google-genai"
        ) from exc

    return ChatGoogleGenerativeAI(
        model=model_name or settings.DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _create_openai(
    settings,
    model_name: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    api_key = _get_secret(settings.OPENAI_API_KEY)
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Install OpenAI support with: "
            "pip install langchain-openai"
        ) from exc

    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _create_groq(
    settings,
    model_name: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    api_key = _get_secret(settings.GROQ_API_KEY)
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "Install Groq support with: pip install langchain-groq"
        ) from exc

    return ChatGroq(
        model_name=model_name or settings.GROQ_MODEL,
        groq_api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _create_ollama(
    settings,
    model_name: Optional[str],
    temperature: float,
    **kwargs,
) -> BaseChatModel:
    """
    Connect to local Ollama via OpenAI-compatible endpoint
    (defaults to http://localhost:11434/v1).
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Install OpenAI/Ollama support with: pip install langchain-openai"
        ) from exc

    base_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"
    return ChatOpenAI(
        model=model_name or settings.OLLAMA_MODEL,
        base_url=base_url,
        api_key="ollama",  # Ollama local endpoint accepts any non-empty string
        temperature=temperature,
        **kwargs,
    )