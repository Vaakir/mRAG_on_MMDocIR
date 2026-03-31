# src/indexing/embedder.py
import torch
from typing import List, Dict, Any
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper for Jina CLIP v2 embedding model (text + image, shared vector space)."""

    def __init__(self, model_name: str = "jinaai/jina-clip-v2"):
        logger.info(f"Loading Jina CLIP v2 model: {model_name}")

        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dimension = 1024  # Jina CLIP v2 outputs 1024 dimensions

        logger.info(f"Jina CLIP v2 loaded. Embedding dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Embed a list of texts using Jina CLIP v2 text encoder."""
        logger.info(f"Embedding {len(texts)} texts with Jina CLIP v2...")
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch_texts,
                prompt_name="document",
                batch_size=8,
            )
            all_embeddings.append(embeddings)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        return self.model.encode([query], prompt_name="retrieval.query")[0]


def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: TextEmbedder
) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks]
    return embedder.embed_texts(texts)
