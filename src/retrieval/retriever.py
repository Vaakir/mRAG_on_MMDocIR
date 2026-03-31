# src/retrieval/retriever.py
# Updated to use Qdrant vector database

from typing import List, Dict, Any
from indexing.embedder import TextEmbedder
from indexing.vector_store import QdrantVectorDB
# -------------------------------------------------------------------
class BaselineRetriever:
    """Simple retriever for the baseline system using Qdrant."""
    #-------------------
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: QdrantVectorDB
    ):
        self.embedder = embedder # Embedder for creating query embeddings
        self.vector_store = vector_store # Qdrant vector store for retrieval
    #-------------------
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
        
        # Retrieve relevant chunks from Qdrant vector store (returns list of dicts with id, score, text, payload)
        results = self.vector_store.retrieve(query_embedding, top_k)
        
        return results
    #-------------------
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve and format context for generation.
        Returns a single string with all retrieved chunks.
        """
        results = self.retrieve(query, top_k) # Get the raw retrieval results (list of dicts with id, score, text, payload)
        
        context_parts = [] # List to hold formatted context parts (each part corresponds to a retrieved chunk)
        
        # Format the retrieved chunks into a single context string (including metadata for better generation)
        for i, result in enumerate(results, 1):
            # Qdrant results have: text, payload (with metadata), score
            pdf_name = result['payload'].get('pdf_name', 'Unknown') # Get the PDF name from metadata (if available)
            text = result['text'] # Get the chunk text from the result
            context_parts.append( # Format each retrieved chunk with its source PDF and text for better context in generation
                f"[Document {i}] (Source: {pdf_name})\n"
                f"{text}\n"
            )
        
        return "\n".join(context_parts)