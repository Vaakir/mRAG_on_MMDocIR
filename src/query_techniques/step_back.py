from typing import Any, Dict, List
from .base import QueryTechnique


class StepBackRetrieval(QueryTechnique):
    """
    Step-Back Retrieval: Generate a simpler, more general version of the question.
    
    Retrieves for both original and step-back questions in parallel,
    then merges and deduplicates results.
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using step-back strategy.
        
        Args:
            question: User's original question
            top_k: Number of top documents to return
            
        Returns:
            List of retrieved documents combining original and step-back results
        """
        # Step 1: Generate step-back (simpler/more general) question
        step_back_question = self._generate_step_back(question)
        
        # Step 2: Retrieve for both original and step-back questions in parallel
        retrieval_sets = self._parallel_retrieve([question, step_back_question], top_k=top_k)
        
        # Step 3: Merge results
        all_results = []
        for results in retrieval_sets:
            all_results.extend(results)
        merged = self._deduplicate_and_rerank(all_results)
        
        return merged[:top_k]

    def _generate_step_back(self, question: str) -> str:
        """
        Generate a step-back version of the question.
        
        Creates a simpler, more general version focusing on core concepts.
        
        Args:
            question: Original question
            
        Returns:
            Simplified step-back version of the question
        """
        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are a question abstraction expert. Generate a simpler, more general version of the given question that focuses on the core concept by removing specific details."
            },
            {
                "role": "user",
                "content": f"""Generate a step-back question that is simpler and more general than the original.
The step-back question should capture the core concept without specific details.

Original question: {question}

Output: Return only the step-back question, nothing else."""
            }
        ])
        
        return response.strip()
