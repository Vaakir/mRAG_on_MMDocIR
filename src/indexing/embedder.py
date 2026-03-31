# src/indexing/embedder.py

from typing import List, Dict, Any
import numpy as np
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
class TextEmbedder:
    """Wrapper for sentence-transformers text embedding model (baseline)."""

    # BGE models require a query prefix for retrieval tasks
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    #-------------------
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading sentence-transformers model: {model_name}")

        self.model = SentenceTransformer(model_name) # Load the embedding model
        self.dimension = self.model.get_sentence_embedding_dimension() # Get the embedding dimension from the model config
        self.is_bge = "bge" in model_name.lower() # Simple check to see if it's a BGE model (for query prefixing)

        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    #-------------------
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
        all_embeddings = [] # List to hold embeddings for all texts

        # Process texts in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size] # Get the current batch of texts
            embeddings = self.model.encode(batch_texts, normalize_embeddings=True) # Get embeddings for the batch (normalized for cosine similarity)
            all_embeddings.append(embeddings) # Append batch embeddings to the list

        return np.vstack(all_embeddings) # Stack all batch embeddings into a single numpy array
    #-------------------
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text (with BGE prefix if applicable)."""
        if self.is_bge:
            query = self.BGE_QUERY_PREFIX + query
        return self.model.encode([query], normalize_embeddings=True)[0]

# -------------------------------------------------------------------
def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedder: TextEmbedder
) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks] # Extract the text from each chunk to create a list of texts for embedding
    return embedder.embed_texts(texts)          # Use the embedder to get embeddings for all the chunk texts
