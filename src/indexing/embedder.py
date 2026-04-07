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
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed a list of texts.
        Returns numpy array of shape (n_texts, embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} texts...")
        all_embeddings = [] # List to hold embeddings for all texts

        # Process texts in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch_texts,
                prompt_name="document",
                batch_size=8,
            )
            all_embeddings.append(embeddings)

    #-------------------
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.vstack(all_embeddings) # Stack all batch embeddings into a single numpy array

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

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            embeddings = self.model.encode(
                batch_paths,
                batch_size=4,
            )
            all_embeddings.append(embeddings)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.vstack(all_embeddings)

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
