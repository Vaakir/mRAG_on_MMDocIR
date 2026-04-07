# src/indexing/embedder.py
import torch
from typing import List, Dict, Any
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
class TextEmbedder:
    """Wrapper for embedding models (supports Jina CLIP v2, BGE, and others)."""

    def __init__(self, model_name: str = "jinaai/jina-clip-v2"):
        logger.info(f"Loading embedding model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.dimension = 1024  # Jina CLIP v2 outputs 1024 dimensions
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}, Device: {self.device}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Embed a list of texts.
        Returns numpy array of shape (n_texts, embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            prompt_name="document",
            batch_size=batch_size,
            show_progress_bar=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return np.array(embeddings)

    def embed_images(
        self,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Embed a list of image file paths using Jina CLIP v2 image encoder.
        Returns numpy array of shape (n_images, embedding_dim).
        Only works with multimodal models (e.g. jinaai/jina-clip-v2).
        """
        logger.info(f"Embedding {len(image_paths)} images...")
        all_embeddings = []

        embeddings = self.model.encode(
            image_paths,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        return self.model.encode([query], prompt_name="retrieval.query", normalize_embeddings=True)[0]

# -------------------------------------------------------------------
def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: TextEmbedder
) -> np.ndarray:
    """Create embeddings for all text chunks."""
    texts = [chunk["text"] for chunk in chunks]
    return embedder.embed_texts(texts)


def create_image_chunk_embeddings(
    image_chunks: List[Dict[str, Any]],
    embedder: TextEmbedder,
) -> np.ndarray:
    """
    Create embeddings for image chunks.
    For page-level chunks: embeds the single image.
    For sliding-window chunks: embeds the first image (representative page).
    """
    # Use the first image path as the representative for each chunk
    image_paths = [chunk["image_paths"][0] for chunk in image_chunks]
    return embedder.embed_images(image_paths)
