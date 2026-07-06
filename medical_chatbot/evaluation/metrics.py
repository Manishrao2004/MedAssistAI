"""
Evaluation Metrics for Retrieval and Response Quality.

Implements Precision@K, Recall@K, Average Similarity Score,
and Average Response Time metrics.
"""

import time
from typing import Any, Callable

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Metrics:
        - Precision@K: Fraction of retrieved results that are relevant.
        - Recall@K: Fraction of relevant items that are retrieved.
        - Average Similarity Score: Mean similarity of top-K results.
        - Average Response Time: Mean end-to-end query latency.
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.results: list[dict[str, Any]] = []
        logger.info("RetrievalEvaluator initialized.")

    @staticmethod
    def precision_at_k(
        retrieved: list[str],
        relevant: list[str],
        k: int = 5,
    ) -> float:
        """
        Compute Precision@K.

        Precision@K = |retrieved_top_k ∩ relevant| / K

        Args:
            retrieved: List of retrieved item identifiers (ordered by rank).
            relevant: List of relevant (ground-truth) item identifiers.
            k: Number of top results to consider.

        Returns:
            Precision@K score between 0.0 and 1.0.
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved: list[str],
        relevant: list[str],
        k: int = 5,
    ) -> float:
        """
        Compute Recall@K.

        Recall@K = |retrieved_top_k ∩ relevant| / |relevant|

        Args:
            retrieved: List of retrieved item identifiers (ordered by rank).
            relevant: List of relevant (ground-truth) item identifiers.
            k: Number of top results to consider.

        Returns:
            Recall@K score between 0.0 and 1.0.
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def average_similarity_score(scores: list[float]) -> float:
        """
        Compute the average similarity score.

        Args:
            scores: List of similarity scores.

        Returns:
            Mean similarity score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def average_response_time(times: list[float]) -> float:
        """
        Compute the average response time.

        Args:
            times: List of response times in seconds.

        Returns:
            Mean response time in seconds.
        """
        if not times:
            return 0.0
        return sum(times) / len(times)

    def run_evaluation(
        self,
        test_queries: list[dict[str, Any]],
        retrieval_fn: Callable,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Run batch evaluation over a set of test queries.

        Each test query dict should contain:
            - 'query': The question string.
            - 'relevant_focus_areas': List of relevant focus_area strings (ground truth).

        Args:
            test_queries: List of test query dicts.
            retrieval_fn: Function that takes a query string and returns
                          a list of RetrievalResult objects.
            k: Number of top results to evaluate.

        Returns:
            Dictionary with aggregated evaluation metrics.
        """
        logger.info("Running evaluation on %d test queries (k=%d) ...", len(test_queries), k)

        all_precision: list[float] = []
        all_recall: list[float] = []
        all_similarity: list[float] = []
        all_times: list[float] = []
        query_results: list[dict[str, Any]] = []

        for i, tq in enumerate(test_queries):
            query = tq["query"]
            relevant = tq.get("relevant_focus_areas", [])

            # Measure retrieval time
            start = time.perf_counter()
            results = retrieval_fn(query)
            elapsed = time.perf_counter() - start

            # Extract retrieved focus areas and scores
            retrieved_areas = [r.focus_area for r in results]
            scores = [r.similarity_score for r in results]

            # Compute metrics
            p_at_k = self.precision_at_k(retrieved_areas, relevant, k)
            r_at_k = self.recall_at_k(retrieved_areas, relevant, k)
            avg_sim = self.average_similarity_score(scores)

            all_precision.append(p_at_k)
            all_recall.append(r_at_k)
            all_similarity.extend(scores)
            all_times.append(elapsed)

            query_results.append({
                "query": query,
                "precision_at_k": p_at_k,
                "recall_at_k": r_at_k,
                "avg_similarity": avg_sim,
                "response_time": elapsed,
                "num_results": len(results),
            })

            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d / %d queries", i + 1, len(test_queries))

        # Aggregate
        evaluation = {
            "num_queries": len(test_queries),
            "k": k,
            "mean_precision_at_k": self.average_similarity_score(all_precision),
            "mean_recall_at_k": self.average_similarity_score(all_recall),
            "mean_similarity_score": self.average_similarity_score(all_similarity),
            "mean_response_time": self.average_response_time(all_times),
            "total_time": sum(all_times),
            "per_query_results": query_results,
        }

        self.results.append(evaluation)
        logger.info(
            "Evaluation complete. P@%d=%.3f, R@%d=%.3f, AvgSim=%.3f, AvgTime=%.3fs",
            k, evaluation["mean_precision_at_k"],
            k, evaluation["mean_recall_at_k"],
            evaluation["mean_similarity_score"],
            evaluation["mean_response_time"],
        )
        return evaluation

    @staticmethod
    def generate_report(results: dict[str, Any]) -> str:
        """
        Generate a formatted evaluation report.

        Args:
            results: Evaluation results dictionary from run_evaluation().

        Returns:
            Formatted report string.
        """
        k = results["k"]
        report_lines = [
            "=" * 60,
            "  RETRIEVAL EVALUATION REPORT",
            "=" * 60,
            "",
            f"  Queries Evaluated : {results['num_queries']}",
            f"  Top-K             : {k}",
            "",
            "  METRICS",
            "  ─────────────────────────────────────────",
            f"  Precision@{k}       : {results['mean_precision_at_k']:.4f}",
            f"  Recall@{k}          : {results['mean_recall_at_k']:.4f}",
            f"  Avg Similarity    : {results['mean_similarity_score']:.4f}",
            f"  Avg Response Time : {results['mean_response_time']:.4f}s",
            f"  Total Time        : {results['total_time']:.2f}s",
            "",
            "=" * 60,
        ]
        return "\n".join(report_lines)
