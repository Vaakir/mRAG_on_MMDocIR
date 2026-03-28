# src/evaluation/generation_metrics.py

from typing import List, Any, Dict
import re
import numpy as np


def normalize_answer(answer: Any) -> str:
    """
    Normalize answer for comparison.
    - Flattens lists to space-separated string
    - Strips markdown formatting (**bold**, *italic*, bullet dashes)
    - Lowercases and strips punctuation
    - Normalizes numbers: removes commas, strips trailing units like 'million'/'billion'
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
    # Strip common unit suffixes after numbers (714.3 million → 714.3)
    text = re.sub(r'(\d+\.?\d*)\s*(million|billion|usd|%)', r'\1', text)
    # Strip punctuation except digits and letters
    text = re.sub(r'[^\w\s\.\-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(prediction: str, ground_truth: Any) -> float:
    """Check if prediction exactly matches ground truth (after normalization)."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def contains_match(prediction: str, ground_truth: Any) -> float:
    """
    Check if the ground truth value appears anywhere in the prediction.
    Useful for cases where the model gives a correct answer with extra explanation.
    For list ground truths: checks if ALL items appear in the prediction.
    """
    pred_norm = normalize_answer(prediction)

    if isinstance(ground_truth, list):
        gt_items = [normalize_answer(item) for item in ground_truth]
        return 1.0 if all(item in pred_norm for item in gt_items) else 0.0
    else:
        gt_norm = normalize_answer(ground_truth)
        return 1.0 if gt_norm in pred_norm else 0.0


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

    return 2 * precision * recall / (precision + recall)


def evaluate_generation(
    predictions: List[str],
    ground_truths: List[Any]
) -> Dict[str, float]:
    """Evaluate generation performance."""
    exact_matches = []
    contains_matches = []
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        exact_matches.append(exact_match(pred, gt))
        contains_matches.append(contains_match(pred, gt))
        f1_scores.append(token_f1(pred, gt))

    return {
        "exact_match": np.mean(exact_matches),
        "contains_match": np.mean(contains_matches),
        "token_f1": np.mean(f1_scores)
    }
