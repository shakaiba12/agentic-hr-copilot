"""
PeopleQuery AI - Core Observability & Tracing Module
Tracks request traces, execution times, metrics, and LangSmith integration.
"""
from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional
from urllib.parse import urlsplit, urlunsplit

try:
    from langsmith import traceable, trace
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger("peoplequery.observability")


_SECRET_KEY_PATTERNS = re.compile(
    r"(?i)(api[_-]?key|password|secret|token|auth|credential|private[_-]?key)"
)


def sanitize_db_url(url: str) -> str:
    """Sanitize database connection string by removing credentials."""
    if not url:
        return ""
    if url.startswith("sqlite:"):
        return url
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc = f"{netloc}:{parts.port}"
            netloc = f"***:***@{netloc}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        pass
    return url


def sanitize_metadata(data: Any) -> Any:
    """Recursively scrub any sensitive key-value pairs from metadata."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if _SECRET_KEY_PATTERNS.search(str(k)):
                cleaned[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                cleaned[k] = sanitize_metadata(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(data, list):
        return [sanitize_metadata(item) for item in data]
    return data


@contextmanager
def time_execution(step_name: str, metadata_dict: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
    """Context manager to measure execution latency of a component."""
    timing_info = {"step": step_name, "duration_ms": 0.0}
    start_time = time.perf_counter()
    try:
        yield timing_info
    finally:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        timing_info["duration_ms"] = elapsed_ms
        if metadata_dict is not None:
            if "timings" not in metadata_dict:
                metadata_dict["timings"] = {}
            metadata_dict["timings"][step_name] = elapsed_ms
        logger.debug(f"Step '{step_name}' completed in {elapsed_ms}ms")


@contextmanager
def safe_trace_span(
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Generator[Optional[Any], None, None]:
    """
    Context manager to trace a child span in LangSmith safely.
    Guarantees that tracing exceptions never break core application flow.
    """
    if not LANGSMITH_AVAILABLE:
        yield None
        return

    clean_inputs = sanitize_metadata(inputs) if inputs else {}
    clean_metadata = sanitize_metadata(metadata) if metadata else {}

    try:
        with trace(
            name=name,
            run_type=run_type,
            inputs=clean_inputs,
            metadata=clean_metadata,
            tags=tags or [],
        ) as run:
            yield run
    except Exception as exc:
        logger.debug(f"LangSmith trace span '{name}' encountered non-fatal error: {exc}")
        yield None
