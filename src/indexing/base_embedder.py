# src/indexing/base_embedder.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseEmbedder(ABC):
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...


def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: BaseEmbedder
) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    return embedder.embed_texts(texts)
