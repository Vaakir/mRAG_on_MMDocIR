# src/indexing/embedder_bge_large.py
from typing import List
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class BgeTextEmbedder(BaseEmbedder):
    """Sentence-transformers embedder — supports BGE and MiniLM models."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.is_bge = "bge" in model_name.lower()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, normalize_embeddings=True)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        if self.is_bge:
            query = self.BGE_QUERY_PREFIX + query
        return self.model.encode([query], normalize_embeddings=True)[0]
