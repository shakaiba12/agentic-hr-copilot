"""
Unit Tests for Pure Mathematical Vector Similarity and Distance Logic.
Tests Cosine Similarity, Distance Conversions, and Edge Cases independently from vector databases.
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b:
        raise ValueError("Vectors cannot be empty.")
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Clamp floating point inaccuracies to [-1.0, 1.0]
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine distance (1 - cosine_similarity)."""
    return 1.0 - cosine_similarity(vec_a, vec_b)


def rank_by_similarity(
    query_vector: List[float],
    candidate_vectors: List[Tuple[str, List[float]]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Rank candidate vectors by descending cosine similarity."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not candidate_vectors:
        return []

    scored = [
        (doc_id, cosine_similarity(query_vector, vec))
        for doc_id, vec in candidate_vectors
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


class TestVectorSimilarityCalculations:
    """Tests for pure vector mathematics."""

    def test_identical_vectors_yield_maximum_similarity(self):
        vec = [0.5, 0.5, 0.5, 0.5]
        sim = cosine_similarity(vec, vec)
        dist = cosine_distance(vec, vec)

        assert math.isclose(sim, 1.0, abs_tol=1e-6)
        assert math.isclose(dist, 0.0, abs_tol=1e-6)

    def test_orthogonal_vectors_yield_zero_similarity(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        dist = cosine_distance(vec_a, vec_b)

        assert math.isclose(sim, 0.0, abs_tol=1e-6)
        assert math.isclose(dist, 1.0, abs_tol=1e-6)

    def test_opposite_vectors_yield_negative_similarity(self):
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(vec_a, vec_b)
        dist = cosine_distance(vec_a, vec_b)

        assert math.isclose(sim, -1.0, abs_tol=1e-6)
        assert math.isclose(dist, 2.0, abs_tol=1e-6)

    def test_zero_vector_handled_gracefully(self):
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec_a, vec_b)
        dist = cosine_distance(vec_a, vec_b)

        assert sim == 0.0
        assert dist == 1.0

    def test_dimension_mismatch_raises_value_error(self):
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            cosine_similarity(vec_a, vec_b)

    def test_empty_vector_raises_value_error(self):
        with pytest.raises(ValueError, match="Vectors cannot be empty"):
            cosine_similarity([], [1.0, 2.0])

    def test_ranking_order_and_top_k(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            ("doc_low", [0.2, 0.8, 0.0]),
            ("doc_exact", [1.0, 0.0, 0.0]),
            ("doc_mid", [0.7, 0.3, 0.0]),
            ("doc_zero", [0.0, 1.0, 0.0]),
        ]

        ranked = rank_by_similarity(query, candidates, top_k=3)

        assert len(ranked) == 3
        assert ranked[0][0] == "doc_exact"
        assert ranked[1][0] == "doc_mid"
        assert ranked[2][0] == "doc_low"
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]

    def test_ranking_empty_candidates(self):
        query = [1.0, 0.0]
        assert rank_by_similarity(query, [], top_k=5) == []

    def test_ranking_invalid_top_k(self):
        query = [1.0, 0.0]
        with pytest.raises(ValueError, match="top_k must be positive"):
            rank_by_similarity(query, [("doc1", [1.0, 0.0])], top_k=0)
