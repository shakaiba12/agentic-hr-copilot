"""
Retrieval Evaluation Metrics for Enterprise RAG.
Implements Recall@K, Mean Reciprocal Rank (MRR), and Precision@K calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Union


def recall_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Calculate Recall@K: Returns 1.0 if at least one ground-truth ID appears in top-K results, else 0.0.
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0

    target_set = set(ground_truth_ids)
    top_k_ids = retrieved_ids[:k]

    return 1.0 if any(item_id in target_set for item_id in top_k_ids) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Union[Set[str], Sequence[str]],
) -> float:
    """
    Calculate Reciprocal Rank (RR): Returns 1 / (rank + 1) for the first relevant item found, else 0.0.
    """
    if not retrieved_ids or not ground_truth_ids:
        return 0.0

    target_set = set(ground_truth_ids)
    for rank, item_id in enumerate(retrieved_ids):
        if item_id in target_set:
            return 1.0 / (rank + 1)

    return 0.0


def precision_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """
    Calculate Precision@K: Fraction of retrieved top-K items that are in ground truth.
    """
    if not retrieved_ids or not ground_truth_ids or k <= 0:
        return 0.0

    target_set = set(ground_truth_ids)
    top_k_ids = retrieved_ids[:k]
    matched = sum(1 for item_id in top_k_ids if item_id in target_set)

    return matched / float(k)


@dataclass
class EvaluationReport:
    """Aggregated retrieval benchmark report across all evaluated queries."""

    queries_evaluated: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_precision_at_5: float
    query_details: List[Dict[str, Any]] = field(default_factory=list)

    def print_summary(self, title: str = "Retrieval Evaluation") -> None:
        """Pretty-print formatted evaluation benchmark metrics."""
        print("========================================")
        print(f" {title}")
        print("========================================")
        print(f"Queries evaluated: {self.queries_evaluated}")
        print("----------------------------------------")
        print(f"Recall@1:          {self.recall_at_1:.4f} ({self.recall_at_1 * 100:.1f}%)")
        print(f"Recall@3:          {self.recall_at_3:.4f} ({self.recall_at_3 * 100:.1f}%)")
        print(f"Recall@5:          {self.recall_at_5:.4f} ({self.recall_at_5 * 100:.1f}%)")
        print(f"Recall@10:         {self.recall_at_10:.4f} ({self.recall_at_10 * 100:.1f}%)")
        print(f"MRR:               {self.mrr:.4f}")
        print(f"Mean Precision@5:  {self.mean_precision_at_5:.4f}")
        print("========================================")
