"""
Evaluation metrics for information retrieval benchmarks.

Computes Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)
for search results against ground-truth document sets.
"""

import json
import re
from pathlib import Path
from typing import Optional

_RE_ARXIV_VERSION = re.compile(r"^(.*?)(v\d+)$")


def normalize_arxiv_id(doc_id: str) -> str:
    """Strip arXiv version suffix (e.g. 1703.08014v2 -> 1703.08014)."""
    if not doc_id:
        return doc_id
    m = _RE_ARXIV_VERSION.match(doc_id)
    return m.group(1) if m else doc_id


def compute_precision_at_k(
    retrieved_ids: list[str], ground_truth_ids: set[str], k: int
) -> float:
    """
    Precision@k: fraction of top-k retrieved documents that are relevant.

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_ids: Set of relevant document IDs for this query.
        k: Consider only the top-k retrieved documents.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    relevant = sum(1 for rid in retrieved_k if normalize_arxiv_id(rid) in ground_truth_ids)
    return relevant / len(retrieved_k)


def compute_recall_at_k(
    retrieved_ids: list[str], ground_truth_ids: set[str], k: int
) -> float:
    """
    Recall@k: fraction of all relevant documents that are found in the top-k.

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_ids: Set of relevant document IDs for this query.
        k: Consider only the top-k retrieved documents.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if k <= 0 or not ground_truth_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    relevant_found = sum(1 for rid in retrieved_k if normalize_arxiv_id(rid) in ground_truth_ids)
    return relevant_found / len(ground_truth_ids)


def compute_mrr(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """
    Mean Reciprocal Rank for a single query.

    MRR = 1 / rank, where rank is the position of the first relevant document
    (1-indexed). Returns 0.0 if no relevant document is found.

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_ids: Set of relevant document IDs for this query.

    Returns:
        Reciprocal rank value between 0.0 and 1.0.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if normalize_arxiv_id(rid) in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def evaluate_single_query(
    results: list[dict],
    ground_truth_ids: set[str],
    ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict:
    """
    Compute all metrics for a single query.

    Args:
        results: Search results list, each dict must have an 'id' key.
        ground_truth_ids: Set of relevant document IDs.
        ks: Values of k for Precision/Recall@k.

    Returns:
        Dict with precision@k, recall@k, and mrr.
    """
    retrieved_ids = [r.get("id", "") for r in results]
    normalized_gt = {normalize_arxiv_id(gid) for gid in ground_truth_ids}
    metrics = {"mrr": compute_mrr(retrieved_ids, normalized_gt)}

    for k in ks:
        metrics[f"precision@{k}"] = compute_precision_at_k(
            retrieved_ids, normalized_gt, k
        )
        metrics[f"recall@{k}"] = compute_recall_at_k(
            retrieved_ids, normalized_gt, k
        )

    return metrics


def aggregate_metrics(
    per_query_metrics: list[dict],
    latencies: Optional[list[float]] = None,
    ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict:
    """
    Compute mean metrics across all queries.

    Args:
        per_query_metrics: List of metric dicts (one per query).
        latencies: Optional list of per-query latencies in milliseconds.
        ks: Values of k for Precision/Recall@k.

    Returns:
        Dict with mean metrics and optional mean latency.
    """
    if not per_query_metrics:
        return {}

    aggregated = {}
    for k in ks:
        aggregated[f"mean_precision@{k}"] = sum(
            m[f"precision@{k}"] for m in per_query_metrics
        ) / len(per_query_metrics)
        aggregated[f"mean_recall@{k}"] = sum(
            m[f"recall@{k}"] for m in per_query_metrics
        ) / len(per_query_metrics)
    aggregated["mean_mrr"] = sum(m["mrr"] for m in per_query_metrics) / len(
        per_query_metrics
    )
    aggregated["total_queries"] = len(per_query_metrics)

    if latencies:
        aggregated["mean_latency_ms"] = sum(latencies) / len(latencies)

    return aggregated


def format_results(mode: str, metrics: dict) -> str:
    """
    Format aggregated metrics as a pretty-printed summary block.

    Args:
        mode: Search mode name (e.g. 'Lexical', 'Hybrid').
        metrics: Aggregated metrics dict from aggregate_metrics().

    Returns:
        Formatted string ready for printing.
    """
    lines = [
        "--- FINAL EVALUATION RESULTS ---",
        f"Mode: {mode}",
        f"Total Queries Evaluated: {metrics.get('total_queries', 0)}",
        f"Mean MRR: {metrics.get('mean_mrr', 0):.4f}",
    ]
    for k in (5, 10, 20, 50):
        lines.append(
            f"Mean Precision@{k}: {metrics.get(f'mean_precision@{k}', 0):.4f}"
        )
    for k in (5, 10, 20, 50):
        lines.append(
            f"Mean Recall@{k}: {metrics.get(f'mean_recall@{k}', 0):.4f}"
        )
    if "mean_latency_ms" in metrics:
        lines.append(
            f"Average End-to-End Latency: {metrics['mean_latency_ms']:.2f} ms"
        )
    return "\n".join(lines)
