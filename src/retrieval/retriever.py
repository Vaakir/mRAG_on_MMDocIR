# src/retrieval/retriever.py
# Updated to use Qdrant vector database

from typing import List, Dict, Any
from indexing.embedder import TextEmbedder
from indexing.vector_store import QdrantVectorDB

class BaselineRetriever:
    """Simple retriever for the baseline system using Qdrant."""
    
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: QdrantVectorDB
    ):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Returns:
            List of dicts with: id, score, text, payload (metadata)
        """
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve from Qdrant
        results = self.vector_store.retrieve(query_embedding, top_k)
        
        return results
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve and format context for generation.
        Returns a single string with all retrieved chunks.
        """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Qdrant results have: text, payload (with metadata), score
            pdf_name = result['payload'].get('pdf_name', 'Unknown')
            text = result['text']
            context_parts.append(
                f"[Document {i}] (Source: {pdf_name})\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)