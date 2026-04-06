from typing import Any, Dict, List
import numpy as np
from .base import QueryTechnique


class HyDERetrieval(QueryTechnique):
    """
    HyDE (Hypothetical Document Embeddings): Generate hypothetical answer documents and use their embeddings.
    
    Generates N hypothetical documents that would answer the question,
    averages their embeddings, and retrieves using the averaged embedding.
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hypothetical document embeddings.
        
        Args:
            question: User's question
            top_k: Number of top documents to return
            
        Returns:
            List of retrieved documents most similar to hypothetical answers
        """
        num_docs = self.config.get('num_variants', 3)
        
        # Step 1: Generate hypothetical documents that would answer the question
        hypothetical_docs = self._generate_hypothetical_docs(question, num_docs)
        
        # Step 2: Embed the hypothetical documents
        doc_embeddings = self.embedder.embed_texts(hypothetical_docs)
        
        # Step 3: Average the embeddings to get a combined search signal
        avg_embedding = np.mean(doc_embeddings, axis=0)
        
        # Step 4: Search using the averaged embedding
        results = self.retriever.retrieve_by_embedding(avg_embedding, top_k=top_k)
        
        return results

    def _generate_hypothetical_docs(self, question: str, num: int) -> List[str]:
        """
        Generate hypothetical documents that would answer the question.
        
        Args:
            question: User's question
            num: Number of hypothetical documents to generate
            
        Returns:
            List of hypothetical document passages
        """
        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are an expert document writer. Generate realistic, informative documents that directly answer the given question. Each document should be well-written and contain relevant details that would appear in a real answer."
            },
            {
                "role": "user",
                "content": f"""Generate exactly {num} hypothetical documents that would contain the answer to this question.
Each document should be a realistic passage that directly answers or relates to the question.
The documents should be detailed and informative.

Question: {question}

Output format: Separate each document with '---' on its own line. Return only the documents."""
            }
        ])
        
        # Parse documents separated by '---'
        docs = [doc.strip() for doc in response.split('---') if doc.strip()]
        return docs[:num]
