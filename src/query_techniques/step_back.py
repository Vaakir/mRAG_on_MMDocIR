from typing import Any, Dict, List
from .base import QueryTechnique
import logging

logger = logging.getLogger(__name__)


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
        "content": """You are a question abstraction expert. Generate a simpler, more general version of the given question that focuses on the core concept by removing specific details.
Here are examples of step-back transformations:
Example 1:
Original: "Among the 42 studies that used BERT models compared in Table 8, how many achieved F1 scores above 0.85?"
Step-back: "What models and their performance metrics are compared in the tables?"
Example 2:
Original: "How many paragraphs in Section 3.2 specifically mention GPU memory optimization techniques while discussing neural network training?"
Step-back: "What are the main technical approaches for neural network training discussed?"
Example 3:
Original: "In the survey of participants aged 25-40 from urban areas, what percentage reported dissatisfaction with service quality in Q2 2023?"
Step-back: "What were the reported satisfaction levels with the service?"
"""
    },
    {
        "role": "user",
        "content": f"""Generate a step-back question that is simpler and more general than the original.
Remove specific details like numbers, names, sections, and qualifiers.
Focus on the core concept.
Original question: {question}
Output: Return only the step-back question, nothing else."""
    }
])
        
        return response.strip()
