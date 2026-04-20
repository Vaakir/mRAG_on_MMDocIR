# src/indexing/embedder_clip.py
import open_clip
import torch
from typing import List
import numpy as np
import logging
from tqdm import tqdm

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class ClipTextEmbedder(BaseEmbedder):
    """OpenCLIP text embedder (ViT-B-32)."""

    def __init__(self, model_name: str = "ViT-B-32"):
        logger.info(f"Loading CLIP model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dimension = 512
        logger.info(f"CLIP model loaded. Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                text_tokens = self.tokenizer(batch_texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features.cpu().numpy())
        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], batch_size=1)[0]
