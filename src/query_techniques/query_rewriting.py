from typing import Any, Dict, List
from .base import QueryTechnique


class QueryRewritingRetrieval(QueryTechnique):
    """
    Query Rewriting: Improve and reformulate questions for better retrieval.
    
    Technique:
    1. Use LLM to rewrite the question for clarity, grammar, and structure
    2. Retrieve documents for both original and rewritten questions in parallel
    3. Merge and deduplicate results from both retrievals
    4. Rerank by relevance score and return top K
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Rewrite question and retrieve using both original and improved formulation.
        
        Args:
            question: User's original question (may be unclear or poorly-phrased)
            top_k: Number of documents to return
            
        Returns:
            List of deduplicated and reranked retrieved documents
        """
        # Step 1: Rewrite the question for clarity and structure
        rewritten_question = self._rewrite_question(question)
        
        # Step 2: Retrieve for both original and rewritten questions in parallel
        retrieval_sets = self._parallel_retrieve([question, rewritten_question], top_k=top_k)
        
        # Step 3: Merge results
        all_results = []
        for results in retrieval_sets:
            all_results.extend(results)
        merged = self._deduplicate_and_rerank(all_results)
        
        return merged[:top_k]

    def _rewrite_question(self, question: str) -> str:
        """
        Improve question formulation for better retrieval.
        
        Rewrites the question to be clearer, more specific, grammatically correct, 
        with expanded acronyms and improved structure for retrieval.
        
        Args:
            question: Original question (may contain grammar issues, unclear references, etc.)
            
        Returns:
            Improved, rewritten version of the question
        """
        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are a query optimization expert. Rewrite questions to be clearer, more specific, and better structured for information retrieval. Fix grammar, expand acronyms, remove noise, and ensure the intent is crystal clear."
            },
            {
                "role": "user",
                "content": f"""Rewrite this question to be clearer and better formulated for information retrieval.
Improve grammar, clarity, structure, and specificity. Fix any acronyms or unclear references.
Do not change the core intent of the question.

Original question: {question}

Output: Return only the rewritten question, nothing else."""
            }
        ])
        
        return response.strip()
