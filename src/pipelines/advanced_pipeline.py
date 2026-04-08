# src/pipelines/advanced_pipeline.py
# Advanced RAG pipeline with query technique support
# Highly modular and configurable for different embedders, LLMs, retrieval methods, etc.

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
    def initialize_components(self):
        """Initialize generator and query technique."""
        super().initialize_components()
        
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
        
        # Generate answer
        answer = self.generator.generate(question, context)
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "num_docs": len(retrieved),
        }
    
    def evaluate(self, test_questions: List[Dict[str, Any]], use_technique: bool = True):
        """
        Evaluate pipeline on test questions (optimized with parallel answer generation).
        
        Two-phase approach:
        1. Retrieve documents for all questions (sequential, with variant generation in parallel)
        2. Generate answers for all questions in parallel (2 workers)
        
        Args:
            test_questions: List of test question dicts with 'question' and 'answer' keys
            use_technique: Use query technique for retrieval
        """
        eval_start_time = time.time()
        logger.info(f"Evaluating on {len(test_questions)} questions with query_technique={use_technique}...")
        
        test_subset = test_questions[:self.config.EVAL_SUBSET_SIZE]
        
        # ===== PHASE 1: Retrieve for all questions in parallel =====
        phase1_start = time.time()
        logger.info(f"Phase 1: Retrieving documents in parallel ({len(test_subset)} questions, {self.config.RETRIEVAL_WORKERS} workers)...")
        all_retrievals = [None] * len(test_subset)
        retrieval_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.RETRIEVAL_WORKERS) as executor:
            # Submit retrieval task for each question
            futures = {}
            for i, test_q in enumerate(test_subset):
                if use_technique and self.query_technique:
                    future = executor.submit(self.retrieve_with_technique, test_q['question'], self.config.TOP_K)
                else:
                    future = executor.submit(self.retrieve, test_q['question'], self.config.TOP_K)
                futures[future] = i
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                i = futures[future]
                retrieved = future.result()
                all_retrievals[i] = retrieved
                
                test_q = test_subset[i]
                retrieval_results.append({
                    'question': test_q['question'],
                    'retrieved': retrieved,
                    'expected_answer': test_q.get('answer', '')
                })
                
                completed += 1
                logger.info(f"  Retrieved {completed}/{len(test_subset)} questions")
        
        logger.info(f"Phase 1 complete. Retrieved for {len(all_retrievals)} questions.")
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 took {phase1_time:.2f} seconds")
        
        # ===== PHASE 2: Generate answers in parallel =====
        phase2_start = time.time()
        logger.info(f"Phase 2: Generating answers in parallel ({len(test_subset)} questions, {self.config.GENERATION_WORKERS} workers)...")
        generation_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.GENERATION_WORKERS) as executor:
            # Submit all generation tasks
            futures = []
            for i, (test_q, retrieved) in enumerate(zip(test_subset, all_retrievals)):
                context = "\n\n".join([
                    f"[Document {j+1}]:\n{result['text']}"
                    for j, result in enumerate(retrieved)
                ])
                future = executor.submit(self.generator.generate, test_q['question'], context)
                futures.append((i, test_q, future))
            
            # Collect results in order as they complete
            completed_count = 0
            for idx, (i, test_q, future) in enumerate(futures):
                answer = future.result()
                completed_count += 1
                logger.info(f"  Generated {completed_count}/{len(futures)} answers...")
                
                print(f"\n{'-'*80}")
                print(f"\nQuestion: {test_q['question']}")
                print(f"Ground Truth: {test_q['answer']}")
                print(f"Generated Answer: {answer}")
                
                generation_results.append({
                    'question': test_q['question'],
                    'answer': answer,
                    'expected_answer': test_q.get('answer', '')
                })
        
        logger.info("Phase 2 complete.")
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 took {phase2_time:.2f} seconds")
        
        # ===== Evaluate metrics =====
        retrieval_metrics = evaluate_retrieval(
            [r['retrieved'] for r in retrieval_results],
            test_subset
        )
        logger.info(f"Retrieval metrics: {retrieval_metrics}")
        
        generation_metrics = evaluate_generation(
            [r['answer'] for r in generation_results],
            [r['expected_answer'] for r in generation_results]
        )
        logger.info(f"Generation metrics: {generation_metrics}")
        
        total_time = time.time() - eval_start_time
        logger.info(f"\nTOTAL EVALUATION TIME: {total_time:.2f}s (Phase 1: {phase1_time:.2f}s + Phase 2: {phase2_time:.2f}s)")
        
        return {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'timing': {
                'total': total_time,
                'phase1': phase1_time,
                'phase2': phase2_time
            }
        }
