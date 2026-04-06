from typing import Any, Dict, List
from .base import QueryTechnique


class QueryDecompositionRetrieval(QueryTechnique):
    """
    Query Decomposition: Break complex questions into focused sub-questions.
    
    Technique:
    1. Decompose original question into N simpler sub-questions
    2. Retrieve documents for original question AND all sub-questions in parallel
    3. Merge all retrieval sets into one combined result set
    4. Deduplicate and rerank by relevance score
    5. Return top K most relevant documents
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Decompose question and retrieve documents for each aspect.
        
        Args:
            question: User's complex question
            top_k: Number of documents to return
            
        Returns:
            List of deduplicated and reranked retrieved documents
        """
        num_subquestions = self.config.get('num_variants', 3)
        
        # Step 1: Decompose the question into sub-questions
        sub_questions = self._decompose_question(question, num_subquestions)
        
        # Step 2: Retrieve for original question + all sub-questions in parallel
        all_queries = [question] + sub_questions
        retrieval_sets = self._parallel_retrieve(all_queries, top_k=top_k)
        
        # Step 3: Merge results
        all_results = []
        for results in retrieval_sets:
            all_results.extend(results)
        
        # Step 4: Deduplicate and return top K
        merged = self._deduplicate_and_rerank(all_results)
        
        return merged[:top_k]

    def _decompose_question(self, question: str, num: int) -> List[str]:
        """
        Break down complex question into simpler sub-questions.
        
        Uses LLM to decompose one complex question into N simpler, focused sub-questions
        that together cover all aspects of the original question.
        
        Args:
            question: Original complex question
            num: Number of sub-questions to generate
            
        Returns:
            List of sub-questions (length <= num)
        """
        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are an expert at decomposing complex questions into simpler, more focused sub-questions that together help answer the original question completely."
            },
            {
                "role": "user",
                "content": f"""Break down this complex question into exactly {num} simpler sub-questions that together cover all aspects of the original question.
Each sub-question should focus on one specific aspect.

Original question: {question}

Output format: One sub-question per line, nothing else. No numbering, no explanations, no extra text."""
            }
        ])
        
        # Parse response (split by newlines, clean up)
        sub_questions = [line.strip() for line in response.split('\n') if line.strip()]
        return sub_questions[:num]
