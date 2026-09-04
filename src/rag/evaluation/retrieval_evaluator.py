"""
Retrieval Benchmark Evaluator for Enterprise RAG.
Runs standardized evaluation over labeled evaluation dataset against ChromaDB vector retriever.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import Settings, get_settings
from src.rag.evaluation.retrieval_metrics import (
    EvaluationReport,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.rag.retrieval.retriever import DenseRetriever


class RetrievalEvaluator:
    """
    Evaluates retrieval pipelines against ground-truth benchmark datasets.
    """

    DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "hr_eval_dataset.json"

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        retriever: Optional[DenseRetriever] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dataset_path = Path(dataset_path or self.DEFAULT_DATASET_PATH)
        self.retriever = retriever or DenseRetriever(settings=self.settings)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")

        content = self.dataset_path.read_text(encoding="utf-8")
        return json.loads(content)

    def evaluate(
        self,
        retriever: Optional[Any] = None,
        top_k: int = 10,
    ) -> EvaluationReport:
        """
        Evaluate retriever against the loaded dataset.

        Args:
            retriever: Retriever instance with .retrieve(query, top_k) method (defaults to self.retriever).
            top_k: Maximum retrieval depth (default: 10).

        Returns:
            EvaluationReport with aggregated benchmark metrics.
        """
        active_retriever = retriever or self.retriever
        dataset = self.load_dataset()

        r1_scores: List[float] = []
        r3_scores: List[float] = []
        r5_scores: List[float] = []
        r10_scores: List[float] = []
        rr_scores: List[float] = []
        p5_scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for sample in dataset:
            query = sample["query"]
            ground_truth = set(sample["relevant_chunk_ids"])

            results = active_retriever.retrieve(query=query, top_k=top_k)
            retrieved_ids = [res.id for res in results]

            r1 = recall_at_k(retrieved_ids, ground_truth, k=1)
            r3 = recall_at_k(retrieved_ids, ground_truth, k=3)
            r5 = recall_at_k(retrieved_ids, ground_truth, k=5)
            r10 = recall_at_k(retrieved_ids, ground_truth, k=10)
            rr = reciprocal_rank(retrieved_ids, ground_truth)
            p5 = precision_at_k(retrieved_ids, ground_truth, k=5)

            r1_scores.append(r1)
            r3_scores.append(r3)
            r5_scores.append(r5)
            r10_scores.append(r10)
            rr_scores.append(rr)
            p5_scores.append(p5)

            details.append({
                "id": sample.get("id"),
                "query": query,
                "ground_truth": list(ground_truth),
                "retrieved_ids": retrieved_ids,
                "recall@5": r5,
                "reciprocal_rank": rr,
            })

        n = max(1, len(dataset))
        report = EvaluationReport(
            queries_evaluated=len(dataset),
            recall_at_1=sum(r1_scores) / n,
            recall_at_3=sum(r3_scores) / n,
            recall_at_5=sum(r5_scores) / n,
            recall_at_10=sum(r10_scores) / n,
            mrr=sum(rr_scores) / n,
            mean_precision_at_5=sum(p5_scores) / n,
            query_details=details,
        )

        return report

    def compare(
        self,
        dense_retriever: Optional[Any] = None,
        reranked_retriever: Optional[Any] = None,
    ) -> Dict[str, EvaluationReport]:
        """
        Compare dense vector search vs two-stage reranked retrieval.
        """
        dense = dense_retriever or self.retriever
        if reranked_retriever is None:
            from src.rag.reranking.pipeline import RerankedRetriever
            reranked = RerankedRetriever(dense_retriever=dense, settings=self.settings)
        else:
            reranked = reranked_retriever

        dense_report = self.evaluate(retriever=dense, top_k=10)
        reranked_report = self.evaluate(retriever=reranked, top_k=5)

        print("\n==========================================================================================")
        print(" RETRIEVAL EVALUATION COMPARISON: VECTOR SEARCH vs CROSS-ENCODER RERANKING")
        print("==========================================================================================")
        print(f"Queries Evaluated: {dense_report.queries_evaluated}")
        print("------------------------------------------------------------------------------------------")
        print(f"{'Pipeline':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<10} | {'Mean P@5':<10}")
        print("------------------------------------------------------------------------------------------")
        print(f"{'Dense Vector Search':<24} | {dense_report.recall_at_1:<10.4f} | {dense_report.recall_at_3:<10.4f} | {dense_report.recall_at_5:<10.4f} | {dense_report.mrr:<10.4f} | {dense_report.mean_precision_at_5:<10.4f}")
        print(f"{'+ Cross-Encoder Rerank':<24} | {reranked_report.recall_at_1:<10.4f} | {reranked_report.recall_at_3:<10.4f} | {reranked_report.recall_at_5:<10.4f} | {reranked_report.mrr:<10.4f} | {reranked_report.mean_precision_at_5:<10.4f}")
        print("==========================================================================================\n")

        return {
            "dense": dense_report,
            "reranked": reranked_report,
        }
