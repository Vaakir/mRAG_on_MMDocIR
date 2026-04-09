# src/evaluation/retrieval_metrics.py
# Comprehensive retrieval evaluation metrics

from typing import List, Dict, Any, Set
import numpy as np
# -------------------------------------------------------------------
def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@k. 
    This measures the proportion of retrieved documents in the top-k that are relevant.
    
    - **Formula: (# relevant docs in top-k) / k**
    - Shows: Basic retrieval quality. Higher is better.
    
    Args:
        retrieved: list of retrieved document identifiers
        relevant: set of relevant document identifiers
        k: number of top documents to consider
    
    Returns:
        Precision@k value (between 0 and 1)
    """
    # Avoid division by zero
    if k == 0:
        return 0.0
    
    # Consider only the top-k retrieved documents
    retrieved_at_k = retrieved[:k]
    
    # Count how many of the top-k retrieved documents are relevant
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    
    return relevant_retrieved / k
# -------------------------------------------------------------------
def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@k.
    This measures the fraction of all relevant documents that appear in the top-k retrieved results.
    
    - **Formula: (# relevant docs in top-k) / (total # relevant docs)**
    - Shows: Coverage of relevant documents. Higher is better. Maximum of 1.0 when all relevant docs are retrieved.
    
    Args:
        retrieved: list of retrieved document identifiers
        relevant: set of relevant document identifiers
        k: number of top documents to consider
    
    Returns:
        Recall@k value (between 0 and 1)
    """
    # Avoid division by zero and empty relevant set
    if len(relevant) == 0:
        return 0.0
    
    # Consider only the top-k retrieved documents
    retrieved_at_k = retrieved[:k]
    
    # Count how many of the top-k retrieved documents are relevant
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    
    # FIXED: Changed from binary hit (0/1) to standard recall formula
    # return 1.0 if any(doc in relevant for doc in retrieved_at_k) else 0.0
    
    return relevant_retrieved / len(relevant)
# -------------------------------------------------------------------
def page_recall_at_k(retrieved: List[Dict[str, Any]], relevant_pdf: str, relevant_pages: Set[int], k: int) -> float:
    """
    Binary hit: 1.0 if any retrieved chunk (top-k) is from the correct PDF
    AND overlaps with at least one ground-truth page_id.
    
    - **Formula: 1.0 if any retrieved chunk in top-k matches relevant PDF and overlaps relevant pages, else 0.0**
    - Shows: Page-level accuracy. Higher is better, but can be 0 or 1 due to binary nature.
    
    Args:
        retrieved: list of retrieved results (dicts with 'payload' containing 'pdf_name' and 'page_numbers')
        relevant_pdf: the PDF name that is relevant for this query
        relevant_pages: set of relevant page IDs
        k: number of top documents to consider
    
    Returns:
        1.0 if any retrieved chunk in top-k matches relevant PDF and overlaps relevant pages, else 0.0
    """
    # Avoid division by zero and empty relevant pages
    if not relevant_pages:
        return 0.0
    
    # Consider only the top-k retrieved results
    for r in retrieved[:k]:
        payload = r.get("payload", {}) # Get the payload from the retrieved result (contains metadata like pdf_name and page_numbers)
        chunk_pdf = payload.get("pdf_name", "") # Get the PDF name from the payload (default to empty string if not found)
        chunk_pages = set(payload.get("page_numbers") or []) # Get the page numbers from the payload and convert to a set (default to empty set if not found)
        if chunk_pdf == relevant_pdf and chunk_pages & relevant_pages: # Check if the chunk's PDF matches the relevant PDF and if there is any overlap between the chunk's pages and the relevant pages
            return 1.0
    
    # If no retrieved chunk in the top-k matches the relevant PDF and overlaps with relevant pages, return 0.0
    return 0.0
# -------------------------------------------------------------------
def mean_average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Mean Average Precision (MAP).
    Measures the average precision at each position where a relevant doc is found.
    
    - **Formula: MAP = (1/# relevant docs) * sum(precision@i for each relevant doc at position i)**
    - Shows: Overall ranking effectiveness. Higher is better, with a maximum of 1.0 when all relevant docs are ranked at the top.
    
    Args:
        retrieved: list of retrieved document identifiers
        relevant: set of relevant document identifiers
    
    Returns:
        MAP value (between 0 and 1)
    """
    # Avoid division by zero
    if len(relevant) == 0:
        return 0.0
    
    score = 0.0 # Cumulative score for MAP
    hits = 0    # Count of relevant documents found so far
    
    # Iterate through the retrieved documents and calculate precision at each position where a relevant document is found
    for i, doc in enumerate(retrieved):
        if doc in relevant: # If the retrieved document is relevant, increment the hit count and add the precision at this position to the score
            hits += 1
            score += hits / (i + 1)  # The precision at position i+1
    
    # Return the average precision by dividing the cumulative score by the total number of relevant documents
    return score / len(relevant)

# -------------------------------------------------------------------
def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (RR).
    Returns 1 / rank of the first relevant document, or 0 if none found.
    
    - **Formula: RR = (1 / rank of first relevant doc) or 0 if none found**
    - Shows: How early the first match appears in the ranking. Higher is better, with a maximum of 1.0 when the first retrieved doc is relevant.
    
    Args:
        retrieved: list of retrieved document identifiers
        relevant: set of relevant document identifiers
    
    Returns:
        1.0 / rank of the first relevant document, or 0 if none found
    """
    # Iterate through the retrieved documents and return the reciprocal (i.e. the inverse; 1/rank) of the rank of the first relevant document found
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1) # Return the reciprocal of the rank (note that it's i+1 because rank is 1-indexed)
    return 0.0

# -------------------------------------------------------------------
def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    Measures ranking quality using discounted gains (better positions weighted higher).
    
    NDCG = DCG@k / IDCG@k
    - DCG@k: Actual ranking quality
    - IDCG@k: Ideal ranking (all relevant docs at top)
    
    - **Formula: NDCG@k = DCG@k / IDCG@k**, where:
        - DCG@k = sum((1 / log2(position+1)) for relevant docs in top-k)
        - IDCG@k = sum((1 / log2(position+1)) for all relevant docs in ideal ranking)
    - Shows: Idea ranking quality (i.e. relevant docs ranked higher). Higher is better, with a maximum of 1.0 when all relevant docs are ranked at the top.
    
    Args:
        retrieved: list of retrieved document identifiers
        relevant: set of relevant document identifiers
        k: number of top documents to consider
        
    Returns:
        NDCG@k value (between 0 and 1)

    """
    # Avoid division by zero and empty relevant set
    if k == 0 or len(relevant) == 0:
        return 0.0
    
    # Consider only the top-k retrieved documents for NDCG calculation
    retrieved_at_k = retrieved[:k] 
    
    # Calculate DCG@k (Discounted Cumulative Gain)
    dcg = 0.0 # Cumulative gain for the actual ranking
    
    # Iterate through the top-k retrieved documents and calculate the gain for each relevant document found, discounted by its position in the ranking
    for i, doc in enumerate(retrieved_at_k):
        if doc in relevant:
            # Standard DCG formula: relevance / log2(position+1)
            # relevance = 1 for binary relevance
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position is 1-indexed, log base 2
    
    # Calculate IDCG@k (Ideal DCG - all relevant docs ranked first)
    ideal_ranking_size = min(len(relevant), k) # The ideal ranking can only have as many relevant documents as exist, but cannot exceed k
    idcg = 0.0 # Cumulative gain for the ideal ranking
    for i in range(ideal_ranking_size): # Iterate through the ideal ranking positions (up to the number of relevant documents or k) and calculate the gain for each position, which is 1 for binary relevance
        idcg += 1.0 / np.log2(i + 2)
    
    # If IDCG is zero (no relevant documents), return 0.0 to avoid division by zero. Otherwise, return the normalized DCG by dividing the actual DCG by the ideal DCG.
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# -------------------------------------------------------------------
def evaluate_retrieval(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance across all queries.
    
    Metrics:
    - Precision@k, Recall@k, Page Recall@k (for k in k_values)
    - MAP: Mean Average Precision (single value across all queries)
    - MRR: Mean Reciprocal Rank (single value across all queries)
    - NDCG@k: Normalized Discounted Cumulative Gain (for k in k_values)

    Args:
        retrieved_results: List of retrieval results per query
        ground_truth: List of ground truth records (with pdf_path and page_ids)
        k_values: List of k values to evaluate (e.g., [1, 3, 5])
    
    Returns:
        Dictionary of averaged metrics across all queries
    """
    metrics = {f"precision@{k}": [] for k in k_values} # Initialize metric lists for each k value
    metrics.update({f"recall@{k}": [] for k in k_values}) # Add recall metrics for each k value
    metrics.update({f"page_recall@{k}": [] for k in k_values}) # Add page recall metrics for each k value
    metrics.update({f"ndcg@{k}": [] for k in k_values}) # Add NDCG metrics for each k value

    # Single-value metrics (averaged separately)
    map_scores = [] # List to hold MAP scores for each query
    mrr_scores = [] # List to hold MRR scores for each query

    # Evaluate each query's retrieval results against the ground truth
    for retrieved, gt in zip(retrieved_results, ground_truth):
        # PDF-level relevant set
        relevant_pdf = gt["pdf_path"].replace("pdf_train/", "").replace("pdf_tests/", "") # Extract the relevant PDF name from the ground truth (removing directory prefixes)
        relevant = {relevant_pdf} # Make a set of relevant PDF names (for precision/recall)

        # Page-level relevant set (ground truth page_ids, 1-indexed)
        relevant_pages = set(gt.get("page_ids") or []) # Make a set of relevant page IDs (for page recall)

        # Retrieved PDF names (for existing precision/recall)
        # Deprecated: (non-deduplicated - includes duplicate PDFs from multiple chunks, may inflate metrics):
        # retrieved_docs = [r["payload"]["pdf_name"] for r in retrieved]
        
        # Deduplicated: doc-level evaluation, preserves order of first occurrence:
        retrieved_docs = list(dict.fromkeys([
            r["payload"].get("pdf_name") or r["payload"].get("doc_name", "unknown")
            for r in retrieved
        ]))

        # Calculate MAP and MRR for this query
        map_scores.append(mean_average_precision(retrieved_docs, relevant)) # Calculate and store the MAP score for this query
        mrr_scores.append(reciprocal_rank(retrieved_docs, relevant))        # Calculate and store the MRR score for this query

        # Calculate Precision@k, Recall@k, Page Recall@k, and NDCG@k for each specified k-value and store the results
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_docs, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_docs, relevant, k))
            metrics[f"page_recall@{k}"].append(page_recall_at_k(retrieved, relevant_pdf, relevant_pages, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved_docs, relevant, k))

    # Average all metrics across queries
    result = {key: np.mean(values) for key, values in metrics.items()}
    
    # Add MAP and MRR (averaged across all queries) to the 'result' dictionary (add after, since they are single-value metrics)
    result["map"] = np.mean(map_scores)
    result["mrr"] = np.mean(mrr_scores)
    
    return result