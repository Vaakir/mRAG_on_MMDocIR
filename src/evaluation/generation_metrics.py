# src/evaluation/generation_metrics.py

from typing import List, Any, Dict, Optional, Tuple
import re
import numpy as np
from collections import Counter
import math
import json
from rouge_score import rouge_scorer


# -------------------------------------------------------------------
def normalize_answer(answer: Any) -> str:
    """
    Normalize answer for comparison. This includes:
    - Flattens lists to space-separated string
    - Strips markdown formatting (**bold**, *italic*, bullet dashes)
    - Lowercases and strips punctuation
    - Normalizes numbers: removes commas, strips trailing units like 'million'/'billion'
    
    Args:
        answer: The answer to normalize (can be string or list)
    
    Returns:
        Normalized string
    """
    if isinstance(answer, list):
        text = " ".join(str(a).lower().strip() for a in answer)
    else:
        text = str(answer).lower().strip()

    # Strip markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Strip bullet points and leading dashes
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    # Remove commas in numbers (1,358,000 → 1358000)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Strip common unit suffixes after numbers (714.3 million --> 714.3)
    text = re.sub(r'(\d+\.?\d*)\s*(million|billion|usd|%)', r'\1', text)
    # Strip punctuation except digits and letters
    text = re.sub(r'[^\w\s\.\-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------------------------------
def exact_match(pred_norm: str, gt_norm: str) -> float:
    """
    Check if prediction exactly matches ground truth (after normalization).
        
    Args:
        pred_norm: The normalized generated answer
        gt_norm: The normalized reference answer
    
    Returns:
        1.0 if exact match, else 0.0
    """
    return 1.0 if pred_norm == gt_norm else 0.0

# -------------------------------------------------------------------
def contains_match(pred_norm: str, gt_norm: str) -> float:
    """
    Check if the ground truth value appears anywhere in the prediction.
    Useful for cases where the model gives a correct answer, but with extra explanation.
    
    Args:
        pred_norm: The normalized generated answer
        gt_norm: The normalized reference answer
    
    Returns:
        1.0 if ground truth is found in prediction, else 0.0
    """
    return 1.0 if gt_norm in pred_norm else 0.0

# -------------------------------------------------------------------
def token_f1(pred_norm: str, gt_norm: str) -> float:
    """
    Calculate token-level F1 score (SQuAD-style, using token frequency).
    
    - **Formula: F1 = 2 * (precision * recall) / (precision + recall)**, where:
        - precision = (number of common tokens) / (number of tokens in prediction)
        - recall = (number of common tokens) / (number of tokens in ground truth)
        - Common tokens are counted respecting frequency (uses Counter intersection)
    
    Args:
        pred_norm: The normalized generated answer
        gt_norm: The normalized reference answer
    
    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0: 
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    precision = sum(common.values()) / len(pred_tokens)
    recall = sum(common.values()) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

# -------------------------------------------------------------------
def rouge_scores(pred_norm: str, gt_norm: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Note: ROUGE is designed for summarization. May not be meaningful for short answers.
    """
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    try:
        scores = scorer.score(gt_norm, pred_norm)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# -------------------------------------------------------------------
def bleu_score(pred_norm: str, gt_norm: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (Bilingual Evaluation Understudy).
    Measures n-gram overlap between prediction and ground truth.
    Useful for summary-like answers and machine translation evaluation.
    
    Args:
        pred_norm: The normalized generated answer
        gt_norm: The normalized reference answer
        max_n: Maximum n-gram size (default 4 for BLEU-4)
    
    Returns:
        BLEU score between 0 and 1
    """
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    precisions = []
    epsilon = 1e-9  # small value for smoothing to avoid zero precision

    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
        )
        gt_ngrams = Counter(
            tuple(gt_tokens[i:i+n]) for i in range(len(gt_tokens) - n + 1)
        )

        if len(pred_ngrams) < n: # if there are no n-grams of this size in the prediction, precision is zero (with smoothing)
            continue
        else:
            matches = sum((pred_ngrams & gt_ngrams).values())
            if matches == 0:
                precisions.append(epsilon)
            else:
                precisions.append(matches / sum(pred_ngrams.values()))
    
    bp = 1.0 if len(pred_tokens) >= len(gt_tokens) else math.exp(1 - len(gt_tokens) / len(pred_tokens))
    
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    return bp * geo_mean

# -------------------------------------------------------------------
def semantic_similarity(pred_norm: str, gt_norm: str, embedder: Optional[Any] = None) -> float:
    """
    Calculate semantic similarity between prediction and ground truth.
    
    If embedder provided: uses embedding-based cosine similarity
    Otherwise: falls back to token F1 (text-based similarity)
    
    Args:
        pred_norm: The normalized generated answer
        gt_norm: The normalized reference answer
        embedder: Optional embedder with embed_texts() method (e.g., from indexing.embedder)
    
    Returns:
        Semantic similarity score between 0 and 1
    """
    if embedder is None:
        return token_f1(pred_norm, gt_norm)
    
    try:
        pred_emb = embedder.embed_texts([pred_norm])[0]
        gt_emb = embedder.embed_texts([gt_norm])[0]
        
        dot_product = np.dot(pred_emb, gt_emb)
        norm_pred = np.linalg.norm(pred_emb)
        norm_gt = np.linalg.norm(gt_emb)
        
        if norm_pred == 0 or norm_gt == 0:
            return 0.0
        
        return float(dot_product / (norm_pred * norm_gt))
    except Exception as e:
        print(f"Warning: Embedding-based similarity failed ({e}), using token F1")
        return token_f1(pred_norm, gt_norm)

# -------------------------------------------------------------------

# -------------------------------------------------------------------
def evaluate_generation(
    predictions: List[str],
    ground_truths: List[Any],
    embedder: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate generation performance with comprehensive metrics.
    
    Args:
        predictions: List of generated answers
        ground_truths: List of reference answers
        embedder: Optional embedder for semantic similarity calculation
    
    Returns:
        Dictionary with all computed metrics
    """
    
    # Strategy pattern (dict of funcs instead of list)
    metrics_fns = {
        "exact_match": exact_match,
        "contains_match": contains_match,
        "token_f1": token_f1,
        "bleu": bleu_score,
        "rouge_scores": rouge_scores,
        "semantic_similarity": lambda p, g: semantic_similarity(p, g, embedder)
    }

    results = {}

    # Evaluate each query's generation results against the ground truth
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        for name, fn in metrics_fns.items():
            out = fn(pred_norm, gt_norm)
            if isinstance(out, dict):
                for k, v in out.items():
                    results.setdefault(k, []).append(v)
            else:
                results.setdefault(name, []).append(out)

    # Average all the metrics across all queries and compile results into a dictionary
    averaged = {}
    for name, scores in results.items():
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            averaged[name] = float(np.mean(valid_scores))
            
    return averaged

