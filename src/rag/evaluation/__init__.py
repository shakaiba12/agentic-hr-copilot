from typing import Any

__all__ = [
    "RetrievalEvaluator",
    "EvaluationReport",
    "recall_at_k",
    "reciprocal_rank",
    "precision_at_k",
]


def __getattr__(name: str) -> Any:
    if name in ("recall_at_k", "reciprocal_rank", "precision_at_k", "EvaluationReport"):
        from src.rag.evaluation.retrieval_metrics import (
            EvaluationReport,
            precision_at_k,
            recall_at_k,
            reciprocal_rank,
        )
        return locals()[name]
    if name == "RetrievalEvaluator":
        from src.rag.evaluation.retrieval_evaluator import RetrievalEvaluator
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
