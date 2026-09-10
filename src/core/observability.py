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
    from langsmith.run_helpers import get_current_run_tree
    from langsmith.run_trees import RunTree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None

    def trace(*args, **kwargs):
        @contextmanager
        def _dummy():
            yield None
        return _dummy()

logger = logging.getLogger("peoplequery.observability")


_SECRET_KEY_PATTERNS = re.compile(
    r"(?i)(api[_-]?key|password|secret|token|auth|credential|private[_-]?key|connection_string)"
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
    """Recursively scrub any sensitive key-value pairs or connection strings from metadata."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            k_str = str(k)
            if _SECRET_KEY_PATTERNS.search(k_str):
                cleaned[k] = "[REDACTED]"
            elif k_str.lower() in ("db_url", "database_url"):
                cleaned[k] = sanitize_db_url(str(v)) if isinstance(v, str) else "[REDACTED]"
            elif isinstance(v, (dict, list)):
                cleaned[k] = sanitize_metadata(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(data, list):
        return [sanitize_metadata(item) for item in data]
    elif isinstance(data, str) and ("postgres://" in data or "postgresql://" in data or "mysql://" in data):
        return sanitize_db_url(data)
    return data


@contextmanager
def time_execution(
    step_name: str,
    metadata_dict: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
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
    clean_tags = list(tags) if tags else []

    try:
        cm = trace(
            name=name,
            run_type=run_type,
            inputs=clean_inputs,
            metadata=clean_metadata,
            tags=clean_tags,
        )
    except Exception as exc:
        logger.debug(f"LangSmith trace span '{name}' initialization encountered non-fatal error: {exc}")
        yield None
        return

    try:
        with cm as run:
            yield run
    except Exception as exc:
        logger.debug(f"LangSmith trace span '{name}' encountered non-fatal error: {exc}")
        # Note: If an unhandled business logic exception was raised inside the with-block, re-raise it
        # so business logic functions correctly, but if it's purely a tracing issue, don't crash.
        raise


def set_trace_metadata(
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
) -> None:
    """Helper to safely attach metadata, tags, inputs, and outputs to the current active trace run."""
    try:
        run = get_current_run_tree()
        if not run:
            return
        if metadata:
            clean_meta = sanitize_metadata(metadata)
            if hasattr(run, "metadata") and isinstance(run.metadata, dict):
                run.metadata.update(clean_meta)
        if tags:
            if hasattr(run, "tags") and isinstance(run.tags, list):
                for t in tags:
                    if t not in run.tags:
                        run.tags.append(t)
        if inputs:
            clean_in = sanitize_metadata(inputs)
            if hasattr(run, "inputs") and isinstance(run.inputs, dict):
                run.inputs.update(clean_in)
            else:
                run.inputs = clean_in
        if outputs:
            clean_out = sanitize_metadata(outputs)
            if hasattr(run, "outputs") and isinstance(run.outputs, dict):
                run.outputs.update(clean_out)
            else:
                run.outputs = clean_out
    except Exception as e:
        logger.debug(f"Could not set trace metadata: {e}")


def record_trace_error(error: Union[str, Exception]) -> None:
    """Helper to record an error on the current active trace run without crashing."""
    try:
        run = get_current_run_tree()
        if not run:
            return
        err_msg = str(error)
        if hasattr(run, "error"):
            run.error = err_msg
        if hasattr(run, "tags") and isinstance(run.tags, list):
            if "error" not in run.tags:
                run.tags.append("error")
        if hasattr(run, "metadata") and isinstance(run.metadata, dict):
            run.metadata["status"] = "error"
            run.metadata["error"] = err_msg
    except Exception as e:
        logger.debug(f"Could not record trace error: {e}")

