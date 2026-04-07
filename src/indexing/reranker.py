# src/indexing/reranker.py
# Cross-encoder reranker: re-scores retrieved chunks by reading query+chunk together.

import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved chunks using a cross-encoder model.

    Stage 1 (BM25 + dense) retrieves top-N candidates quickly.
    Stage 2 (this) scores each query-chunk pair together and returns top-k.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder reranker loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return top_k.

        Args:
            query: The user query string
            candidates: Retrieved chunks (each must have a 'text' field)
            top_k: Number of results to return after reranking

        Returns:
            Top-k chunks sorted by cross-encoder score (descending)
        """
        if not candidates:
            return candidates

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
