# src/indexing/hybrid_retriever.py
# Hybrid retrieval: BM25 (keyword) + Dense (semantic) with Reciprocal Rank Fusion

import logging
import re
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'\w+', text.lower())


class HybridRetriever:
    """
    Combines BM25 keyword retrieval with dense vector retrieval using
    Reciprocal Rank Fusion (RRF).

    BM25 excels at exact keyword matches (company names, codes, IDs).
    Dense retrieval excels at semantic similarity.
    RRF combines both rankings without needing score calibration.
    """

    RRF_K = 60  # RRF constant — controls rank smoothing

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder,
        vector_db,
        top_k: int = 5,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_db = vector_db
        self.top_k = top_k

        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        n_candidates = min(k * 20, len(self.chunks))  # fetch more for RRF

        # --- Dense retrieval ---
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_db.retrieve(query_emb, top_k=n_candidates)
        # Map chunk id → dense rank (0-based)
        dense_rank = {r["id"]: rank for rank, r in enumerate(dense_results)}

        # --- BM25 retrieval ---
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_top_ids = np.argsort(bm25_scores)[::-1][:n_candidates].tolist()
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_top_ids)}

        # --- Reciprocal Rank Fusion ---
        all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        rrf_scores = {}
        for doc_id in all_ids:
            dr = dense_rank.get(doc_id, n_candidates)
            br = bm25_rank.get(doc_id, n_candidates)
            rrf_scores[doc_id] = 1 / (self.RRF_K + dr) + 1 / (self.RRF_K + br)

        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        # --- Build result list ---
        dense_by_id = {r["id"]: r for r in dense_results}
        results = []
        for doc_id in top_ids:
            if doc_id in dense_by_id:
                entry = {**dense_by_id[doc_id], "score": rrf_scores[doc_id]}
            else:
                chunk = self.chunks[doc_id]
                entry = {
                    "id": doc_id,
                    "score": rrf_scores[doc_id],
                    "text": chunk["text"],
                    "payload": {
                        "pdf_name": chunk["pdf_name"],
                        "pdf_path": chunk["pdf_path"],
                        "chunk_id": chunk["chunk_id"],
                        "char_len": chunk["char_len"],
                    },
                }
            results.append(entry)

        return results
    
    def retrieve_by_embedding(self, embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using a pre-computed embedding (useful for HyDE and other embedding-based techniques).
        
        Args:
            embedding: Pre-computed embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents
        """
        k = top_k or self.top_k
        
        # Dense retrieval only (no BM25 for embedding-based search)
        dense_results = self.vector_db.retrieve(embedding, top_k=k)
        
        return dense_results
