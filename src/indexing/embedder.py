# src/indexing/embedder.py
import threading
import torch
import threading
from typing import List, Dict, Any
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
class TextEmbedder:
    """Wrapper for embedding models (supports Jina CLIP v2, BGE, and others)."""

    def __init__(self, model_name: str = "jinaai/jina-clip-v2"):
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dimension = 1024  # Jina CLIP v2 outputs 1024 dimensions
        self._lock = threading.Lock()  # jina-clip rotary embedding is not thread-safe

        # logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        print(f"\n[OK] Embedding model loaded. Dimension: {self.dimension}")

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
            # Normalize text to lowercase for consistent tokenization
            batch_texts = [text.lower() for text in texts[i:i + batch_size]]
            
            # Thread-safe encoding
            with self.encode_lock:
                embeddings = self.model.encode(
                    batch_texts,
                    prompt_name="document",
                    batch_size=8,
                    normalize_embeddings=True
                )
            all_embeddings.append(embeddings)

    #-------------------
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.vstack(all_embeddings) # Stack all batch embeddings into a single numpy array

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        with self._lock:
            return self.model.encode([query], prompt_name="retrieval.query", normalize_embeddings=True)[0]

    def embed_images(
        self,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Embed a list of image paths using the CLIP image encoder.
        Returns numpy array of shape (n_images, embedding_dim).
        """
        logger.info(f"Embedding {len(image_paths)} images...")
        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            embeddings = self.model.encode(
                images,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            all_embeddings.append(embeddings)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.vstack(all_embeddings)

# -------------------------------------------------------------------
def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: TextEmbedder
) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks] # Extract the text from each chunk to create a list of texts for embedding
    return embedder.embed_texts(texts)          # Use the embedder to get embeddings for all the chunk texts
