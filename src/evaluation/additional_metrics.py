# src/evaluation/additional_metrics.py
"""
Additional generation metrics (BLEU, ROUGE, Semantic Similarity)
These are supplementary to the required metrics in generation_metrics.py
"""

from typing import List, Any, Dict
import re
import numpy as np

# Optional imports (install if needed)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed. BLEU scores will not be available.")
    print("Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. ROUGE scores will not be available.")
    print("Install with: pip install rouge-score")

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Semantic similarity not available.")
    print("Install with: pip install sentence-transformers")


def normalize_answer(answer: Any) -> str:
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        return " ".join(str(a).lower().strip() for a in answer)
    return str(answer).lower().strip()


def bleu_score(prediction: str, ground_truth: Any) -> float:
    """
    Calculate BLEU score.
    Note: BLEU is designed for longer texts. May not be meaningful for short answers.
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    
    # Tokenize
    pred_tokens = pred_normalized.split()
    gt_tokens = gt_normalized.split()
    
    # Handle edge cases
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # BLEU expects reference as list of token lists
    reference = [gt_tokens]
    
    # Use smoothing for short texts
    smoothing = SmoothingFunction().method1
    
    try:
        score = sentence_bleu(reference, pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0


def rouge_scores(prediction: str, ground_truth: Any) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Note: ROUGE is designed for summarization. May not be meaningful for short answers.
    """
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    try:
        scores = scorer.score(gt_normalized, pred_normalized)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def semantic_similarity(
    prediction: str, 
    ground_truth: Any,
    model: SentenceTransformer = None
) -> float:
    """
    Calculate semantic similarity using sentence embeddings.
    This is REQUIRED by the project and more meaningful than BLEU/ROUGE for short answers.
    """
    if not SBERT_AVAILABLE:
        return 0.0
    
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    
    # Handle edge cases
    if len(pred_normalized) == 0 or len(gt_normalized) == 0:
        return 0.0
    
    # Load model if not provided (cache for reuse)
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
    
    try:
        # Encode both texts
        pred_embedding = model.encode(pred_normalized, convert_to_tensor=True)
        gt_embedding = model.encode(gt_normalized, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(pred_embedding, gt_embedding).item()
        
        # Normalize to [0, 1] (cosine is [-1, 1])
        return (similarity + 1) / 2
    except:
        return 0.0


def evaluate_additional_metrics(
    predictions: List[str],
    ground_truths: List[Any]
) -> Dict[str, float]:
    """
    Evaluate generation with additional metrics.
    
    Returns:
        Dictionary with average scores for:
        - bleu: BLEU score (if available)
        - rouge1, rouge2, rougeL: ROUGE scores (if available)
        - semantic_similarity: Embedding-based similarity (REQUIRED by project)
    """
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    semantic_scores = []
    
    # Load semantic similarity model once for all predictions
    sbert_model = None
    if SBERT_AVAILABLE:
        try:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            pass
    
    for pred, gt in zip(predictions, ground_truths):
        # BLEU
        if NLTK_AVAILABLE:
            bleu_scores.append(bleu_score(pred, gt))
        
        # ROUGE
        if ROUGE_AVAILABLE:
            rouge = rouge_scores(pred, gt)
            rouge1_scores.append(rouge['rouge1'])
            rouge2_scores.append(rouge['rouge2'])
            rougeL_scores.append(rouge['rougeL'])
        
        # Semantic Similarity (REQUIRED)
        if SBERT_AVAILABLE:
            semantic_scores.append(semantic_similarity(pred, gt, sbert_model))
    
    results = {}
    
    if bleu_scores:
        results['bleu'] = np.mean(bleu_scores)
    
    if rouge1_scores:
        results['rouge1'] = np.mean(rouge1_scores)
        results['rouge2'] = np.mean(rouge2_scores)
        results['rougeL'] = np.mean(rougeL_scores)
    
    if semantic_scores:
        results['semantic_similarity'] = np.mean(semantic_scores)
    
    return results


# Example usage
if __name__ == "__main__":
    # Test with short factual answers (typical for your dataset)
    predictions = ["8", "Nike Inc.", "The company reported $50M revenue in 2020"]
    ground_truths = ["8", "Nike", "In 2020, the company's revenue was $50 million"]
    
    print("Testing additional metrics:")
    print("="*60)
    
    for pred, gt in zip(predictions, ground_truths):
        print(f"\nPrediction: {pred}")
        print(f"Ground Truth: {gt}")
        print("-"*60)
        
        if NLTK_AVAILABLE:
            print(f"BLEU: {bleu_score(pred, gt):.4f}")
        
        if ROUGE_AVAILABLE:
            rouge = rouge_scores(pred, gt)
            print(f"ROUGE-1: {rouge['rouge1']:.4f}")
            print(f"ROUGE-L: {rouge['rougeL']:.4f}")
        
        if SBERT_AVAILABLE:
            print(f"Semantic Similarity: {semantic_similarity(pred, gt):.4f}")
    
    print("\n" + "="*60)
    print("Aggregate metrics:")
    metrics = evaluate_additional_metrics(predictions, ground_truths)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
