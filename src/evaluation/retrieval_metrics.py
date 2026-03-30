# src/evaluation/retrieval_metrics.py

from typing import List, Dict, Any, Set
import numpy as np

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@k.
    retrieved: list of retrieved document identifiers
    relevant: set of relevant document identifiers
    """
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_retrieved / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@k as a binary hit: 1.0 if any relevant doc appears in top-k, else 0.0.
    This avoids recall > 1 when multiple chunks from the same PDF are retrieved.
    """
    if len(relevant) == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    return 1.0 if any(doc in relevant for doc in retrieved_at_k) else 0.0

def page_recall_at_k(retrieved: List[Dict[str, Any]], relevant_pdf: str, relevant_pages: Set[int], k: int) -> float:
    """
    Binary hit: 1.0 if any retrieved chunk (top-k) is from the correct PDF
    AND overlaps with at least one ground-truth page_id.
    """
    if not relevant_pages:
        return 0.0
    for r in retrieved[:k]:
        payload = r.get("payload", {})
        chunk_pdf = payload.get("pdf_name", "")
        chunk_pages = set(payload.get("page_numbers") or [])
        if chunk_pdf == relevant_pdf and chunk_pages & relevant_pages:
            return 1.0
    return 0.0

def evaluate_retrieval(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance across all queries.

    retrieved_results: List of retrieval results per query
    ground_truth: List of ground truth records (with pdf_path and page_ids)
    """
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"page_recall@{k}": [] for k in k_values})

    for retrieved, gt in zip(retrieved_results, ground_truth):
        # PDF-level relevant set
        relevant_pdf = gt["pdf_path"].replace("pdf_train/", "").replace("pdf_tests/", "")
        relevant = {relevant_pdf}

        # Page-level relevant set (ground truth page_ids, 1-indexed)
        relevant_pages = set(gt.get("page_ids") or [])

        # Retrieved PDF names (for existing precision/recall)
        retrieved_docs = [r["payload"]["pdf_name"] for r in retrieved]

        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_docs, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_docs, relevant, k))
            metrics[f"page_recall@{k}"].append(page_recall_at_k(retrieved, relevant_pdf, relevant_pages, k))

    # Average across all queries
    return {key: np.mean(values) for key, values in metrics.items()}