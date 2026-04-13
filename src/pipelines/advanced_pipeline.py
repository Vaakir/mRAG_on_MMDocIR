# src/pipelines/advanced_pipeline.py
# Advanced RAG pipeline with query technique support
# Highly modular and configurable for different embedders, LLMs, retrieval methods, etc.

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from generation.prompts import get_prompt_strategy
from config.config import AdvancedConfig
from pipelines.base_pipeline import BaseRAGPipeline
from query_techniques import get_query_technique
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

logger = logging.getLogger(__name__)

class AdvancedRAGPipeline(BaseRAGPipeline):
    """
    Advanced RAG pipeline with configurable components and query techniques.
    Inherits initialization and retrieval logic from BaseRAGPipeline.
    """
    
    def __init__(self, config: 'AdvancedConfig'):
        super().__init__(config)
        self.query_technique = None
        self.prompt_strategy = None
        
    def initialize_components(self):
        """Initialize generator, prompting strategy, and query technique."""
        super().initialize_components()
        
        logger.info(f"Initializing prompting strategy: {self.config.PROMPTING_STRATEGY}")
        strategy_config = self.config.PROMPTING_STRATEGY_CONFIG.copy()
        
        # Pass embedder to ensemble if using embedding_similarity aggregation
        if (self.config.PROMPTING_STRATEGY == 'ensemble' and 
            strategy_config.get('aggregation_method') == 'embedding_similarity'):
            logger.info("Ensemble using embedding_similarity - providing embedder to strategy")
            strategy_config['embedder'] = self.embedder
        
        self.prompt_strategy = get_prompt_strategy(
            self.config.PROMPTING_STRATEGY,
            self.generator,
            strategy_config
        )
        
        logger.info(f"Initializing query technique: {self.config.QUERY_TECHNIQUE}")
        self.query_technique = get_query_technique(
            self.config.QUERY_TECHNIQUE,
            self.embedder,
            self.hybrid_retriever or getattr(self, 'retriever', None),
            self.generator,
            self.config.QUERY_TECHNIQUE_CONFIG
        )
    
    def retrieve_with_technique(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using the selected query technique.
        
        Args:
            question: User's question
            top_k: Override default top_k if needed
            
        Returns:
            List of retrieved documents from query technique
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        if self.query_technique is None:
            raise RuntimeError("Query technique not initialized. Call initialize_components() first.")
        
        return self.query_technique.retrieve(question, top_k)
    
    def run_query(self, question: str, use_technique: bool = True, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single query through the pipeline.
        
        Args:
            question: User's question
            use_technique: Use query technique for retrieval (True) or basic retrieval (False)
            top_k: Override default top_k if needed
            
        Returns:
            Dictionary with question, retrieved context, and answer
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        # Retrieve documents
        if use_technique and self.query_technique:
            retrieved = self.retrieve_with_technique(question, top_k)
        else:
            retrieved = self.retrieve(question, top_k)
        
        # Format context
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # Generate answer using the configured prompting strategy
        answer = self.prompt_strategy.generate(question, context)
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "num_docs": len(retrieved),
        }
    
