from typing import Any, Dict, List
from .multi_query import MultiQueryRetrieval


class RAGFusionRetrieval(MultiQueryRetrieval):
    """
    RAG-Fusion: Multi-Query Retrieval with Reciprocal Rank Fusion scoring.
    
    Technique:
    1. Generate N paraphrases of original question (inherits from MultiQueryRetrieval)
    2. Retrieve documents for original question AND all paraphrases in parallel
    3. Apply Reciprocal Rank Fusion (RRF) to combine rankings without score calibration
    4. Score documents by their combined rank across all retrieval sets: score = sum(1/(K+rank)) for all mentions
    5. Return top K by RRF score
    """

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Generate paraphrases and retrieve using RRF ranking fusion.
        
        Args:
            question: User's question
            top_k: Number of documents to return
            
        Returns:
            List of documents ranked by Reciprocal Rank Fusion score
        """
        num_queries = self.config.get('num_queries', 3)

        # Step 1: Generate paraphrases
        paraphrases = self._generate_paraphrases(question, num_queries)
        
        # Step 2: Retrieve for original question + all paraphrases in parallel
        all_queries = [question] + paraphrases
        retrieval_sets = self._parallel_retrieve(all_queries, top_k=top_k)
        
        # Step 3: Apply RRF and return top K
        merged = self._rrf_and_rerank(retrieval_sets)
        
        return merged[:top_k]

    def _rrf_and_rerank(self, retrieval_sets: List[List[Dict]]) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine multiple ranking sets.
        
        RRF combines rankings from multiple queries without requiring score calibration.
        Each document is scored by the sum of 1/(K+rank) for each query it appears in.
        
        Args:
            retrieval_sets: List of retrieval result sets (original question + paraphrases)
            
        Returns:
            List of documents sorted by RRF score (highest first)
        """
        RRF_K = 60  # Constant for RRF formula
        chunk_rrf_scores = {}
        
        # For each retrieval set (original query + all paraphrases)
        for retrieval_set in retrieval_sets:
            for rank, result in enumerate(retrieval_set, start=1):
                chunk_id = result['id']
                rrf_score = 1.0 / (RRF_K + rank)
                
                if chunk_id not in chunk_rrf_scores:
                    chunk_rrf_scores[chunk_id] = {
                        'result': result,
                        'rrf_score': 0.0
                    }
                
                chunk_rrf_scores[chunk_id]['rrf_score'] += rrf_score
        
        # Update results with RRF scores and sort
        merged_results = []
        for chunk_id, data in chunk_rrf_scores.items():
            result = data['result'].copy()
            result['score'] = data['rrf_score']
            merged_results.append(result)
        
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        return merged_results
