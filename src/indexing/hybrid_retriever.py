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
        allowed_types: List[str] = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_db = vector_db
        self.top_k = top_k
        self.allowed_types = allowed_types or ["text"]  # e.g. ["text"] — prevents page_images from consuming candidate slots

        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

        # Lookup for context expansion: (pdf_name, chunk_id) → flat index
        self._chunk_lookup: Dict[tuple, int] = {
            (c["pdf_name"], c["chunk_id"]): i
            for i, c in enumerate(self.chunks)
            if "pdf_name" in c and "chunk_id" in c
        }

    @staticmethod
    def _strip_doc_prefix(text: str) -> str:
        """Remove the [Document: X] prefix that chunk_loader prepends."""
        if text.startswith("[Document:"):
            newline = text.find("\n")
            if newline != -1:
                return text[newline + 1:]
        return text

    def _expand_context(self, results: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
        """
        For each result, prepend/append up to `window` adjacent chunks from the
        same document. The core retrieved text is preserved under 'core_text'.
        Neighboring chunks have their [Document:] prefix stripped to avoid
        repeating it mid-context.
        """
        expanded = []
        for result in results:
            payload = result.get("payload", {})
            pdf_name = payload.get("pdf_name")
            chunk_id = payload.get("chunk_id")

            if pdf_name is None or chunk_id is None:
                expanded.append(result)
                continue

            before, after = [], []
            for offset in range(-window, 0):
                idx = self._chunk_lookup.get((pdf_name, chunk_id + offset))
                if idx is not None:
                    before.append(self._strip_doc_prefix(self.chunks[idx]["text"]))

            for offset in range(1, window + 1):
                idx = self._chunk_lookup.get((pdf_name, chunk_id + offset))
                if idx is not None:
                    after.append(self._strip_doc_prefix(self.chunks[idx]["text"]))

            if before or after:
                context = " ".join(before + [result["text"]] + after)
                expanded.append({**result, "text": context, "core_text": result["text"]})
            else:
                expanded.append(result)

        return expanded

    def retrieve(self, query: str, top_k: int = None, context_window: int = 0) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        n_candidates = min(k * 20, len(self.chunks))  # fetch more for RRF

        # --- Dense retrieval ---
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_db.retrieve(query_emb, top_k=n_candidates, allowed_types=self.allowed_types)
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

        if context_window > 0:
            results = self._expand_context(results, context_window)

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
        dense_results = self.vector_db.retrieve(embedding, top_k=k,  allowed_types=self.allowed_types)
        
        return dense_results
