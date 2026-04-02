# src/evaluation/generation_metrics.py

from typing import List, Any, Dict, Optional, Tuple
import re
import numpy as np
from collections import Counter
import math
import json
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
    # If the answer is a list, join it into a single string (since some answers may be lists of items)
    if isinstance(answer, list):
        text = " ".join(str(a).lower().strip() for a in answer)
    else: # If it's a single string, just normalize it directly
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
def exact_match(prediction: str, ground_truth: Any) -> float:
    """
    Check if prediction exactly matches ground truth (after normalization).
        
    Args:
        prediction: The generated answer to evaluate
        ground_truth: The reference answer (can be string or list)
    
    Returns:
        1.0 if exact match, else 0.0
    """
    pred_norm = normalize_answer(prediction) # Normalize the prediction for comparison
    gt_norm = normalize_answer(ground_truth) # Normalize the ground truth for comparison
    return 1.0 if pred_norm == gt_norm else 0.0 # Return 1.0 if the normalized prediction exactly matches the normalized ground truth, otherwise return 0.0

# -------------------------------------------------------------------
def contains_match(prediction: str, ground_truth: Any) -> float:
    """
    Check if the ground truth value appears anywhere in the prediction.
    Useful for cases where the model gives a correct answer, but with extra explanation.
    For list ground truths: checks if ALL items appear in the prediction.
    
    Args:
        prediction: The generated answer to evaluate
        ground_truth: The reference answer (can be string or list)
    
    Returns:
        1.0 if all items in ground truth are found in prediction, else 0.0

    """
    pred_norm = normalize_answer(prediction) # Normalize the prediction for comparison

    if isinstance(ground_truth, list): # If ground truth is a list, check if all items in it are contained/present in the prediction
        gt_items = [normalize_answer(item) for item in ground_truth] # Normalize each item in the list
        return 1.0 if all(item in pred_norm for item in gt_items) else 0.0 # Return 1.0 only if all items in the list are found in the prediction, otherwise 0.0
    else:
        gt_norm = normalize_answer(ground_truth) # Normalize the ground truth directly if it's not a list
        return 1.0 if gt_norm in pred_norm else 0.0 # Return 1.0 if the normalized ground truth is found anywhere in the normalized prediction, otherwise 0.0

# -------------------------------------------------------------------
def token_f1(prediction: str, ground_truth: Any) -> float:
    """
    Calculate token-level F1 score.
    
    - **Formula: F1 = 2 * (precision * recall) / (precision + recall)**, where:
        - precision = (number of common tokens) / (number of tokens in prediction)
        - recall = (number of common tokens) / (number of tokens in ground truth)
    
    Args:
        prediction: The generated answer to evaluate
        ground_truth: The reference answer (can be string or list)
    
    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = set(normalize_answer(prediction).split()) # Tokenize the normalized prediction into a set of unique tokens for comparison
    gt_tokens = set(normalize_answer(ground_truth).split()) # Tokenize the normalized ground truth into a set of unique tokens for comparison

    if len(pred_tokens) == 0 and len(gt_tokens) == 0: # If both prediction and ground truth have no tokens (empty), consider it a perfect match with F1 of 1.0
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:  # If one of them is empty and the other is not, then there are no common tokens, so F1 is 0.0
        return 0.0

    common = pred_tokens & gt_tokens # Find the set of common tokens between prediction and ground truth
    precision = len(common) / len(pred_tokens) # Calculate precision as the ratio of common tokens to total number of tokens in the prediction
    recall = len(common) / len(gt_tokens) # Calculate recall as the ratio of common tokens to total number of tokens in the ground truth

    if precision + recall == 0: # If both precision and recall are zero (no common tokens), return F1 as 0.0 to avoid division by zero
        return 0.0

    # Calculate F1 score for this prediction-ground truth pair
    return 2 * precision * recall / (precision + recall)

# -------------------------------------------------------------------
def bleu_score(prediction: str, ground_truth: Any, max_n: int = 4) -> float:
    """
    Calculate BLEU score (Bilingual Evaluation Understudy).
    Measures n-gram overlap between prediction and ground truth.
    Useful for summary-like answers and machine translation evaluation.
    
    Args:
        prediction: Generated answer
        ground_truth: Ground truth answer (string or list)
        max_n: Maximum n-gram size (default 4 for BLEU-4)
    
    Returns:
        BLEU score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split() # Tokenize the normalized prediction into a list of tokens for n-gram analysis
    gt_tokens = normalize_answer(ground_truth).split() # Tokenize the normalized ground truth into a list of tokens for n-gram analysis
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0: # If either prediction or ground truth has no tokens, then there can be no n-gram overlap, so return BLEU score of 0.0
        return 0.0
    
    # Calculate n-gram precisions
    precisions = [] # List to hold precision scores for each n-gram size (from 1 to max_n)
    
    # For each n-gram size, calculate the precision of n-grams in the prediction that match n-grams in the ground truth
    for n in range(1, max_n + 1):
        pred_ngrams = Counter( # Create a Counter of n-grams in the prediction (tuples of tokens) for the current n-gram size
            tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
        )
        gt_ngrams = Counter( # Create a Counter of n-grams in the ground truth (tuples of tokens) for the current n-gram size
            tuple(gt_tokens[i:i+n]) for i in range(len(gt_tokens) - n + 1)
        )
        
        if len(pred_ngrams) == 0: # If there are no n-grams of this size in the prediction, then precision is 0.0 for this n-gram size (since we can't have any matches)
            precisions.append(0.0)
        else: # Otherwise, calculate the number of matching n-grams (i.e. the intersection of prediction and ground truth n-grams) and divide by total n-grams in the prediction to get the precision for this n-gram size
            matches = sum((pred_ngrams & gt_ngrams).values()) # Count the number of n-grams in the prediction that also appear in the ground truth (the intersection of the two Counters)
            precisions.append(matches / sum(pred_ngrams.values())) # Precision is calculated as the ratio of matching n-grams to total n-grams in the prediction for this particular n-gram size
    
    # Brevity penalty (BP) applied to penalize short predictions
    bp = 1.0 if len(pred_tokens) >= len(gt_tokens) else math.exp(1 - len(gt_tokens) / len(pred_tokens))
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions): # If any precision is 0, the geometric mean is 0
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions)) # Calculate the geometric mean of the precision scores for all n-gram sizes (using logarithms to avoid underflow)
    
    return bp * geo_mean # Final BLEU score is the product of the BP and the geometric mean of the n-gram precisions

# -------------------------------------------------------------------
def semantic_similarity(prediction: str, ground_truth: Any, embedder: Optional[Any] = None) -> float:
    """
    Calculate semantic similarity between prediction and ground truth.
    
    If embedder provided: uses embedding-based cosine similarity (LLM-based)
    Otherwise: falls back to token F1 (text-based similarity)
    
    Args:
        prediction: Generated answer
        ground_truth: Ground truth answer
        embedder: Optional embedder with embed_texts() method (e.g., from indexing.embedder)
    
    Returns:
        Semantic similarity score between 0 and 1
    """
    # If no embedder is provided, use token F1 as a fallback similarity measure (since we can't compute embeddings without an embedder)
    if embedder is None:
        # Fallback to token F1 as similarity measure
        return token_f1(prediction, ground_truth)
    
    try:
        # Use embeddings to compute cosine similarity
        pred_emb = embedder.embed_texts([prediction])[0] # Embed the prediction using the provided embedder (assuming it has an embed_texts method that takes a list of texts and returns a list of embeddings, we take the first embedding since we only have one prediction)
        gt_emb = embedder.embed_texts([normalize_answer(ground_truth)])[0] # Embed the normalized ground truth using the same embedder (normalizing it first to ensure consistency in embedding) and take the first embedding since we only have one ground truth answer
        
        # Calculate Cosine similarity
        dot_product = np.dot(pred_emb, gt_emb) # Dot product between the prediction embedding and the ground truth embedding
        norm_pred = np.linalg.norm(pred_emb)   # L2 norm of the prediction embedding
        norm_gt = np.linalg.norm(gt_emb)       # L2 norm of the ground truth embedding
        
        # If either embedding has zero norm (which can happen if the text is empty or very short), we can't compute cosine similarity, so we return 0.0 in that case to indicate no similarity
        if norm_pred == 0 or norm_gt == 0:
            return 0.0
        
        # Otherwise, return the cosine similarity score (dot product divided by the product of the norms of the two embeddings). This gives a score between -1 and 1, but since we are dealing with text embeddings, we expect it to be between 0 and 1 for similar texts.
        return float(dot_product / (norm_pred * norm_gt))
    except Exception as e: # If embedding-based similarity fails for any reason (e.g., embedder error, empty text causing zero norm), we catch the exception and fall back to using token F1 as a simpler similarity measure that doesn't rely on embeddings
        print(f"Warning: Embedding-based similarity failed ({e}), using token F1")
        return token_f1(prediction, ground_truth)

# -------------------------------------------------------------------
# def accuracy(prediction: str, ground_truth: Any) -> float:
#     """
#     Binary accuracy: 1.0 if prediction is exactly correct (after normalization), 0.0 otherwise.
#     Similar to exact_match but named differently for clarity.
    
#     Args:
#         prediction: The generated answer to evaluate
#         ground_truth: The reference answer (can be string or list)
    
#     Returns:
#         1.0 if exact match, else 0.0
#     """
#     # Accuracy is essentially the same as exact match for generation evaluation, since we want to know if the prediction is correct or not. We can reuse the exact_match function for this purpose.
#     return exact_match(prediction, ground_truth)
# -------------------------------------------------------------------
def faithfulness_prompt() -> str:
    """
    Return the system prompt for evaluating faithfulness of a generated answer.
    Faithfulness checks if the answer is grounded in the provided context.
    By "grounded" we mean that all factual claims in the answer are supported by or 
    derivable from the context, and that the answer does not contradict the context or 
    introduce facts that are not found in the context.
    """
    return """You are an expert evaluator of text generation systems. Your task is to assess the FAITHFULNESS of a generated answer.

FAITHFULNESS measures whether the generated answer is grounded in the provided context. An answer is faithful if:
1. All factual claims in the answer are supported by or derivable from the context
2. The answer does not contradict the context
3. The answer does not introduce facts not found in the context

You will be given:
- A CONTEXT (retrieved documents)
- A GENERATED ANSWER
- A REFERENCE ANSWER (ground truth)

Output a JSON with:
{
    "score": <float between 0 and 1>,
    "explanation": "<brief explanation of why this score>",
    "issues": ["<list of unfaithful claims if any>"]
}

Be strict: only give a score above 0.7 if the answer is clearly grounded in the context."""

# NOTE: The faithfulness evaluation relies on the LLM evaluator's ability to understand 
# the context and assess the generated answer against it. The prompt is designed to guide 
# the LLM to focus on factual grounding and consistency with the context when assigning 
# a faithfulness score.
# -------------------------------------------------------------------
def evaluate_generation(
    predictions: List[str],
    ground_truths: List[Any],
    contexts: Optional[List[str]] = None,
    embedder: Optional[Any] = None,
    llm_evaluator: Optional[Any] = None
) -> Dict[str, float]:
    """
    Evaluate generation performance with comprehensive metrics.
    
    Args:
        predictions: List of generated answers
        ground_truths: List of reference answers
        contexts: Optional list of retrieved contexts for each query
        embedder: Optional embedder for semantic similarity calculation
        llm_evaluator: Optional LLM evaluator for faithfulness scoring
                      (should have generate() method taking (prompt, context) and returning JSON)
    
    Returns:
        Dictionary with all computed metrics
    """
    exact_matches = []        # List to hold 'exact match' scores
    contains_matches = []     # List to hold 'contains match' scores
    f1_scores = []            # List to hold token 'F1' scores
    bleu_scores = []          # List to hold 'BLEU' scores
    semantic_sims = []        # List to hold 'semantic similarity' scores
    faithfulness_scores = []  # List to hold 'faithfulness' scores (if computed)

    # Evaluate each query's generation results against the ground truth
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        exact_matches.append(exact_match(pred, gt)) # Calculating exact match and appending result to list
        contains_matches.append(contains_match(pred, gt)) # Calculating contains match and appending result to list
        f1_scores.append(token_f1(pred, gt)) # Calculating token F1 score and appending result to list
        bleu_scores.append(bleu_score(pred, gt)) # Calculating BLEU score and appending result to list
        
        # Calculating semantic similarity (with embedder, if available)
        semantic_sims.append(semantic_similarity(pred, gt, embedder))
        
        # Faithfulness evaluation (if context and LLM evaluator available)
        if contexts and llm_evaluator and i < len(contexts):
            try:
                context = contexts[i] # Get the retrieved context for this query (if available)
                prompt = faithfulness_prompt() # Get the system prompt for faithfulness evaluation
                evaluation = llm_evaluator.generate( # Use the LLM evaluator to generate a faithfulness score based on the prompt, context, generated answer, and reference answer
                    prompt,
                    f"CONTEXT:\n{context}\n\nGENERATED ANSWER:\n{pred}\n\nREFERENCE ANSWER:\n{gt}"
                )
                # Parse JSON response to extract score                
                eval_json = json.loads(evaluation) # Parse the LLM's response as JSON to extract the faithfulness score
                faithfulness_scores.append(eval_json.get("score", 0.5)) # Append the faithfulness score to the list (default to 0.5 if not provided)
            except Exception as e: # If there is an error during faithfulness evaluation (e.g., LLM error, JSON parsing error), we catch the exception and append None to the faithfulness scores to indicate that it couldn't be computed for this sample
                print(f"Warning: Faithfulness evaluation failed for sample {i}: {e}")
                faithfulness_scores.append(None)

    # Average all the metrics across all queries and compile results into a dictionary
    result = {
        "exact_match": np.mean(exact_matches),
        "contains_match": np.mean(contains_matches),
        "token_f1": np.mean(f1_scores),
        "bleu": np.mean(bleu_scores),
        "semantic_similarity": np.mean(semantic_sims),
    }
    
    # Add faithfulness, if computed
    if faithfulness_scores and any(s is not None for s in faithfulness_scores): # Only include faithfulness in the results if we have at least one valid score (i.e., not all are None)
        valid_scores = [s for s in faithfulness_scores if s is not None] # Filter out None values from the faithfulness scores to compute the average only on valid scores
        if valid_scores: # If there are valid scores, calculate the mean and add it to the result dictionary
            result["faithfulness"] = np.mean(valid_scores)
    
    return result
