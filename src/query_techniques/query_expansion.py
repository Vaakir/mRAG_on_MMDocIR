from typing import Any, Dict, List
from .base import QueryTechnique


class QueryExpansionRetrieval(QueryTechnique):
    """
    Query Expansion: Add synonyms and related terms to broaden retrieval.
    
    Technique:
    1. Generate N expanded versions of the question using synonyms and related terms
    2. Retrieve documents for original question AND all expanded variants in parallel
    3. Merge all retrieval sets into one combined result set
    4. Deduplicate and rerank by relevance score
    5. Return top K most relevant documents
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Expand question and retrieve using original and synonymous formulations.
        
        Args:
            question: User's original question
            top_k: Number of documents to return
            
        Returns:
            List of deduplicated and reranked retrieved documents
        """
        num_expansions = self.config.get('num_variants', 3)
        
        # Step 1: Generate expanded versions with synonyms/related terms
        expanded_queries = self._expand_query(question, num_expansions)
        
        # Step 2: Retrieve for original question + all expansions in parallel
        all_queries = [question] + expanded_queries
        retrieval_sets = self._parallel_retrieve(all_queries, top_k=top_k)
        
        # Step 3: Merge results
        all_results = []
        for results in retrieval_sets:
            all_results.extend(results)
        
        # Step 4: Merge and deduplicate
        merged = self._deduplicate_and_rerank(all_results)
        
        return merged[:top_k]

    def _expand_query(self, question: str, num: int) -> List[str]:
        """
        Generate expanded versions of the question using synonyms and related terms.
        
        Creates N alternative formulations that express the same core meaning using
        different terminology, synonyms, and related concepts.
        
        Args:
            question: Original question
            num: Number of expanded versions to generate
            
        Returns:
            List of expanded question variants (length <= num)
        """
        response = self.generator.chat([
            {
                "role": "system",
                "content": "You are a query expansion expert. Generate alternative phrasings of questions using synonyms, related terms, and broader/narrower concepts to capture different ways someone might search for the same information."
            },
            {
                "role": "user",
                "content": f"""Generate exactly {num} expanded versions of this question using synonyms and related terms.
Each expanded version should use different words but convey the same core meaning.
Include alternative terminology, related concepts, and different phrasings.

Original question: {question}

Output format: One expanded query per line, nothing else. No numbering, no explanations, no extra text."""
            }
        ])
        
        # Parse response (split by newlines, clean up)
        expanded_queries = [line.strip() for line in response.split('\n') if line.strip()]
        return expanded_queries[:num]
