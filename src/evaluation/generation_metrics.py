# src/evaluation/generation_metrics.py

from typing import List, Any, Dict
import re
import numpy as np

def normalize_answer(answer: Any) -> str:
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        return " ".join(str(a).lower().strip() for a in answer)
    return str(answer).lower().strip()

def exact_match(prediction: str, ground_truth: Any) -> float:
    """Check if prediction exactly matches ground truth."""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    return 1.0 if pred_normalized == gt_normalized else 0.0

def token_f1(prediction: str, ground_truth: Any) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())
    
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_generation(
    predictions: List[str],
    ground_truths: List[Any]
) -> Dict[str, float]:
    """Evaluate generation performance."""
    exact_matches = []
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        exact_matches.append(exact_match(pred, gt))
        f1_scores.append(token_f1(pred, gt))
    
    return {
        "exact_match": np.mean(exact_matches),
        "token_f1": np.mean(f1_scores)
    }