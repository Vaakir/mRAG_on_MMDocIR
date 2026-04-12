# src/pipelines/baseline_pipeline.py
# Updated to use team's integrated components:
# - unstructured for PDF processing
# - Team's Chunking class with fixed_size strategy
# - Qdrant vector database
# - Jina CLIP v2 for embeddings (1024D, multimodal text + image shared space)
# - Ollama API for generation (qwen3:32b for strong reasoning)

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from config.config import BaselineConfig
from pipelines.base_pipeline import BaseRAGPipeline
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
class BaselineRAGPipeline(BaseRAGPipeline):
    """
    Complete baseline RAG pipeline using team's integrated components.
    Inherits initialization and retrieval logic from BaseRAGPipeline.
    """
    
    def __init__(self, config=None):
        super().__init__(config or BaselineConfig())
        
    def run_query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Run a single query through the pipeline with timing instrumentation.
        
        Tracks:
        - Retrieval time
        - Generation time
        - Total query time
        
        Returns:
            Dict with question, retrieved_docs, context, answer, and timing info
        """
        top_k = top_k or self.config.TOP_K
        query_total_start = time.time()

        
        # ===== STEP 1: RETRIEVAL =====
        retrieval_start = time.time()
        retrieved = self.retrieve(question, top_k)
        retrieval_time = time.time() - retrieval_start
        
        # ===== STEP 2: CONTEXT FORMATTING =====
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # ===== STEP 3: GENERATION =====
        generation_start = time.time()
        answer = self.generator.generate(question, context)
        generation_time = time.time() - generation_start
        
        query_total_time = time.time() - query_total_start
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": query_total_time
            }
        }
    #-------------------
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data with comprehensive timing instrumentation.
        """
        logger.info(f"Evaluating on {len(test_data)} test queries...\n")
        
        eval_total_start = time.time()
        
        all_retrieved = []
        all_predictions = []
        all_ground_truths = []
        
        # ===== PHASE 1: RETRIEVAL & GENERATION =====
        phase1_start = time.time()
        
        for i, record in enumerate(test_data, 1):
            logger.info(f"Processing query {i}/{len(test_data)}")
            result = self.run_query(record["question"])

            logger.info(f"Question: {record['question']}")
            logger.info(f"Ground Truth: {record['answer']}")
            logger.info(f"Generated Answer: {result['answer']}\n")

            all_retrieved.append(result['retrieved_docs'])
            all_predictions.append(result['answer'])
            all_ground_truths.append(record['answer'])
        
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 (Retrieval + Generation): {phase1_time:.2f}s")
        
        # ===== PHASE 2: METRIC COMPUTATION =====
        phase2_start = time.time()
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_data)
        generation_metrics = evaluate_generation(
            all_predictions, 
            all_ground_truths,
            embedder=self.embedder  # Pass embedder for true semantic similarity (Jina embeddings)
        )
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        
        eval_total_time = time.time() - eval_total_start
        
        all_metrics = {
            **retrieval_metrics,
            **generation_metrics,
            'timing': {
                'phase1': phase1_time,
                'phase2': phase2_time,
                'total': eval_total_time
            }
        }
        
        print("\n === EVALUATION RESULTS ===")
        for metric_key, metric_value in all_metrics.items():
            if metric_key != 'timing':
                if isinstance(metric_value, (int, float)):
                    print(f"{metric_key}: {metric_value:.4f}")
                else:
                    print(f"{metric_key}: {metric_value}")
                    
        print("\n === TIMING BREAKDOWN ===")
        print(f"Phase 1 (Query Processing): {phase1_time:.2f}s")
        print(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        print(f"Total Evaluation Time: {eval_total_time:.2f}s")
        
        return all_metrics