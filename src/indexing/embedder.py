# src/indexing/embedder.py

import torch
from typing import List, Dict, Any
import numpy as np
import logging
from tqdm import tqdm

# Try OpenAI CLIP first, fall back to open_clip
try:
    import clip
    USE_OPENAI_CLIP = True
except ImportError:
    try:
        import open_clip
        USE_OPENAI_CLIP = False
    except ImportError:
        raise ImportError(
            "Neither 'clip' nor 'open_clip' found. "
            "Install one: pip install clip or pip install open_clip_torch"
        )

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper for CLIP embedding model (text encoder for baseline)."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        logger.info(f"Loading CLIP model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        if USE_OPENAI_CLIP:
            self.model, self.preprocess = clip.load(
                model_name, 
                device=self.device
            )
            self.model.eval()
            self.dimension = 512  # ViT-B/32 outputs 512 dimensions
        else:
            # Use open_clip as fallback
            model_name_ocv = (
                "ViT-B-32" if model_name == "ViT-B/32" else model_name
            )
            self.model, _, self.preprocess = (
                open_clip.create_model_and_transforms(
                    model_name_ocv,
                    pretrained="openai",
                    device=self.device
                )
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name_ocv)
            self.dimension = 512  # ViT-B-32 outputs 512 dimensions
        
        logger.info(
            f"CLIP model loaded. Embedding dimension: {self.dimension}"
        )
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed a list of texts using CLIP text encoder.
        Returns numpy array of shape (n_texts, embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} texts with CLIP...")
        all_embeddings = []
        
        # Process in batches
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size), 
                desc="Encoding texts"
            ):
                batch_texts = texts[i:i + batch_size]
                
                if USE_OPENAI_CLIP:
                    # OpenAI CLIP
                    text_tokens = clip.tokenize(
                        batch_texts, 
                        truncate=True
                    ).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                else:
                    # open_clip
                    text_tokens = self.tokenizer(batch_texts).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                
                # Normalize embeddings (important for cosine similarity)
                text_features = (
                    text_features / 
                    text_features.norm(dim=-1, keepdim=True)
                )
                
                # Move to CPU and convert to numpy
                all_embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        embeddings = self.embed_texts([query], batch_size=1)
        return embeddings[0]


def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: TextEmbedder
) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)
    return embeddings
