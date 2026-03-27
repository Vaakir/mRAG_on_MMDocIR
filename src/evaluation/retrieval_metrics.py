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
    Calculate Recall@k.
    """
    if len(relevant) == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_retrieved / len(relevant)

def evaluate_retrieval(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance across all queries.
    
    retrieved_results: List of retrieval results per query
    ground_truth: List of ground truth records (with pdf_path)
    """
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    
    for retrieved, gt in zip(retrieved_results, ground_truth):
        # Get relevant document (the source PDF)
        relevant_pdf = gt["pdf_path"].replace("pdf_train/", "").replace("pdf_tests/", "")
        relevant = {relevant_pdf}
        
        # Get retrieved document names (pdf_name is inside payload)
        retrieved_docs = [r["payload"]["pdf_name"] for r in retrieved]
        
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_docs, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_docs, relevant, k))
    
    # Average across all queries
    return {key: np.mean(values) for key, values in metrics.items()}