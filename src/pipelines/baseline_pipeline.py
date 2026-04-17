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
from generation.prompts.standard import StandardPromptStrategy
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
        # Baseline ALWAYS uses the standard (direct extraction) strategy
        self.prompt_strategy = StandardPromptStrategy(generator=self.generator)
        
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
        system_prompt = self.prompt_strategy.get_system_prompt()
        answer = self.generator.generate(question, context, system_prompt)
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
