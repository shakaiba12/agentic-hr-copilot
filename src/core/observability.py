"""
PeopleQuery AI - Core Observability & Tracing Module
Tracks request traces, execution times, and metrics.
"""
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("peoplequery.observability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


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
