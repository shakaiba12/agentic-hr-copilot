"""
Unit Tests for Retrieval Evaluation Layer (Phase 3).
Verifies Recall@K, MRR, Precision@K, and Dataset-driven Evaluation.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import List

from src.rag.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.rag.evaluation.retrieval_metrics import (
    EvaluationReport,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.rag.retrieval.retriever import RetrievalResult


class TestRetrievalMetrics:
    """Unit tests for metric functions."""

    def test_recall_at_k_hit_in_top_k(self):
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
        ground_truth = {"doc_b"}

        assert recall_at_k(retrieved, ground_truth, k=1) == 0.0
        assert recall_at_k(retrieved, ground_truth, k=2) == 1.0
        assert recall_at_k(retrieved, ground_truth, k=4) == 1.0

    def test_recall_at_k_miss(self):
        retrieved = ["doc_a", "doc_b", "doc_c"]
        ground_truth = {"doc_z"}

        assert recall_at_k(retrieved, ground_truth, k=3) == 0.0

    def test_recall_at_k_empty_inputs(self):
        assert recall_at_k([], {"doc_a"}, k=5) == 0.0
        assert recall_at_k(["doc_a"], set(), k=5) == 0.0
        assert recall_at_k(["doc_a"], {"doc_a"}, k=0) == 0.0

    def test_reciprocal_rank(self):
        ground_truth = {"target_doc"}

        assert reciprocal_rank(["target_doc", "doc_b"], ground_truth) == 1.0  # Rank 1 -> 1/1 = 1.0
        assert reciprocal_rank(["doc_a", "target_doc"], ground_truth) == 0.5  # Rank 2 -> 1/2 = 0.5
        assert reciprocal_rank(["doc_a", "doc_b", "target_doc"], ground_truth) == 1.0 / 3.0  # Rank 3 -> 1/3
        assert reciprocal_rank(["doc_a", "doc_b"], ground_truth) == 0.0  # Not found -> 0.0

    def test_precision_at_k(self):
        retrieved = ["rel_1", "unrel_1", "rel_2", "unrel_2"]
        ground_truth = {"rel_1", "rel_2", "rel_3"}

        assert precision_at_k(retrieved, ground_truth, k=1) == 1.0  # 1/1
        assert precision_at_k(retrieved, ground_truth, k=2) == 0.5  # 1/2
        assert precision_at_k(retrieved, ground_truth, k=4) == 0.5  # 2/4


class MockRetriever:
    """Mock retriever returning deterministic canned results."""

    def __init__(self, mapping: dict[str, List[str]]):
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        ids = self.mapping.get(query, [])[:top_k]
        return [
            RetrievalResult(id=item_id, text=f"Text for {item_id}", metadata={}, distance=0.1 * (i + 1))
            for i, item_id in enumerate(ids)
        ]


class TestRetrievalEvaluator:
    """Unit tests for RetrievalEvaluator runner."""

    def test_evaluator_with_mock_data(self, tmp_path: Path):
        dataset = [
            {"id": "q1", "query": "query one", "relevant_chunk_ids": ["c1"]},
            {"id": "q2", "query": "query two", "relevant_chunk_ids": ["c2"]},
        ]
        dataset_file = tmp_path / "test_dataset.json"
        dataset_file.write_text(json.dumps(dataset), encoding="utf-8")

        mock_retriever = MockRetriever({
            "query one": ["c1", "other_1"],
            "query two": ["wrong_1", "c2"],
        })

        evaluator = RetrievalEvaluator(dataset_path=dataset_file)
        report: EvaluationReport = evaluator.evaluate(retriever=mock_retriever, top_k=5)

        assert report.queries_evaluated == 2
        # q1 hit at rank 1 (r1=1, rr=1.0)
        # q2 hit at rank 2 (r1=0, r3=1, rr=0.5)
        assert report.recall_at_1 == 0.5
        assert report.recall_at_3 == 1.0
        assert report.recall_at_5 == 1.0
        assert report.mrr == (1.0 + 0.5) / 2.0  # 0.75
