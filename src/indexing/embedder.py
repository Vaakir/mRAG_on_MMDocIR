# src/indexing/embedder.py
import threading
import torch
from typing import List
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class TextEmbedder(BaseEmbedder):
    """Jina CLIP v2 embedder — shared text/image vector space."""

    def __init__(self, model_name: str = "jinaai/jina-clip-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.dimension = 1024
        self._lock = threading.Lock()
        print(f"\n[OK] Embedding model loaded. Dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = [text.lower() for text in texts[i:i + batch_size]]
            with self._lock:
                embeddings = self.model.encode(
                    batch_texts,
                    prompt_name="document",
                    batch_size=8,
                    normalize_embeddings=True,
                )
            all_embeddings.append(embeddings)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            return self.model.encode(
                [query], prompt_name="retrieval.query", normalize_embeddings=True
            )[0]

    def embed_images(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []

        is_jina = "jina" in (getattr(self.model, "model_card_data", None) and getattr(self.model.model_card_data, "model_name", "") or str(getattr(self.model, "tokenizer", "")))

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            with self._lock:
                if is_jina and hasattr(self.model[0], 'auto_model') and hasattr(self.model[0].auto_model, 'encode_image'):
                    auto_model = self.model[0].auto_model
                    embeddings = auto_model.encode_image(images)
                    if hasattr(embeddings, 'detach'):
                        embeddings = embeddings.detach().cpu().numpy()
                elif is_jina and hasattr(self.model[0], 'model') and hasattr(self.model[0].model, 'encode_image'):
                    auto_model = self.model[0].model
                    embeddings = auto_model.encode_image(images)
                    if hasattr(embeddings, 'detach'):
                        embeddings = embeddings.detach().cpu().numpy()
                elif is_jina and hasattr(self.model._first_module(), 'encode_image'):
                    # Workaround for jina-clip-v2 custom module in newer SentenceTransformers
                    embeddings = self.model._first_module().encode_image(images)
                    if hasattr(embeddings, 'detach'):
                        embeddings = embeddings.detach().cpu().numpy()
                else:
                    embeddings = self.model.encode(
                        images,
                        normalize_embeddings=True,
                        batch_size=batch_size,
                    )
            
            all_embeddings.append(embeddings)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.vstack(all_embeddings)

