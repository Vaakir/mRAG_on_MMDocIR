from abc import ABC, abstractmethod
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class QueryTechnique(ABC):
    """Abstract base - all techniques inherit from this"""
    
    def __init__(self, embedder, retriever, generator, config=None):
        # Each subclass gets access to these tools
        self.embedder = embedder           # Has: embed_query(), embed_texts()
        self.retriever = retriever         # Has: retrieve(question) -> List[Dict]
        self.generator = generator         # Has: chat(), generate()
        self.config = config or {}         # Dict with technique-specific settings
    
    @abstractmethod
    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Each subclass MUST implement this.
        
        Args:
            question: User's question string
            top_k: Number of top results to return
        
        Returns:
            List of retrieved chunks, each chunk should be:
            {
                'id': chunk_id,
                'text': chunk_text,
                'score': relevance_score,
                'metadata': {...}
            }
        """
        pass

    def _deduplicate_and_rerank(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks, keep highest-scoring occurrence"""
        chunk_dict = {}
        
        for result in results:
            chunk_id = result['id']
            
            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = result.copy()
            else:
                if result['score'] > chunk_dict[chunk_id]['score']:
                    chunk_dict[chunk_id] = result.copy()
        
        # Sort by score descending
        merged_results = list(chunk_dict.values())
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        
        return merged_results
    
    def _parallel_retrieve(self, queries: List[str], top_k: int = 5, max_workers: int = 4) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries in parallel.
        
        Useful for techniques like multi_query, step_back, query_decomposition, etc.
        that need to retrieve multiple times and can benefit from parallelization.
        
        Args:
            queries: List of query strings to retrieve for
            top_k: Number of top results per query
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of result lists, one per query (same order as input queries)
        """
        # Use minimum of number of queries and max_workers (cap at 4 to avoid excessive threads)
        num_workers = min(len(queries), max_workers)
        results_list = [None] * len(queries)  # Maintain order
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all retrieval tasks
            future_to_idx = {
                executor.submit(self.retriever.retrieve, query, top_k): idx
                for idx, query in enumerate(queries)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_list[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error retrieving query {idx}: {e}")
                    results_list[idx] = []
        
        return results_list