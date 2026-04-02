from typing import Any, Dict, List
from .base import QueryTechnique


class MultiQueryRetrieval(QueryTechnique):
    """
    Multi-Query Retrieval: Generate paraphrases and retrieve for all variants.
    
    Generates N paraphrases, retrieves for original + paraphrases in parallel,
    then merges and deduplicates results.
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multi-query strategy.
        
        Args:
            question: User's original question
            top_k: Number of top documents to return
            
        Returns:
            List of retrieved documents, deduplicated and ranked by score
        """
        num_queries = self.config.get('num_variants', 3)

        # Step 1: Generate paraphrases using the generator (LLM)
        paraphrases = self._generate_paraphrases(question, num_queries)
        # Returns: ["What's the sky color?", "Describe sky colors?", "Sky appearance?"]
        
        # Step 2: Retrieve for original + all paraphrases in parallel
        all_queries = [question] + paraphrases
        retrieval_sets = self._parallel_retrieve(all_queries, top_k=top_k)
        
        # Step 3: Merge all results
        all_results = []
        for results in retrieval_sets:
            all_results.extend(results)
        
        # Step 4: Deduplicate and return top K
        merged = self._deduplicate_and_rerank(all_results)
        
        return merged[:top_k]



    def _generate_paraphrases(self, question: str, num: int) -> List[str]:
        """
        Generate paraphrases of the question using an LLM.
        
        Args:
            question: Original question string
            num: Number of paraphrases to generate
            
        Returns:
            List of paraphrased questions
        """

        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are a question paraphrasing expert. Generate clear, grammatically correct alternative phrasings that preserve the original meaning and intent."
            },
            {
                "role": "user",
                "content": f"""Generate exactly {num} paraphrases of this question. Each paraphrase should be a different way to ask the same thing.

Output format: One paraphrase per line, nothing else. No numbering, no explanations, no extra text.

Original question: {question}"""
            }
        ])

        # Parse response (split by newlines, clean up)
        paraphrases = [line.strip() for line in response.split('\n') if line.strip()]
        return paraphrases[:num]
