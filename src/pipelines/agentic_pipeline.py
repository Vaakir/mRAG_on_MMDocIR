"""System 3: Agentic RAG Pipeline - extends BaseRAGPipeline with agent orchestration"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.config import AdvancedConfig
from pipelines.base_pipeline import BaseRAGPipeline
from agentic.graph.builder import build_agentic_graph
from agentic.graph.state import AgenticRAGState
from query_techniques import get_query_technique

logger = logging.getLogger(__name__)


class AgenticRAGPipeline(BaseRAGPipeline):
    """
    Agentic RAG Pipeline: Orchestrates agents for query rewriting, retrieval, 
    grading, and generation.
    
    Extends BaseRAGPipeline to reuse index building and component initialization.
    """
    
    def __init__(self, config: AdvancedConfig):
        """Initialize the agentic pipeline."""
        super().__init__(config)
        self.agentic_graph = None
        self.llm = None
        
    def initialize_components(self):
        """Initialize all components including LLM for agent decision-making."""
        super().initialize_components()
        
        # Initialize LLM for agent decision-making
        from agentic.llm import SimpleLLM
        
        logger.info(f"Initializing LLM for agent decisions: {self.config.LLM_MODEL}")
        
        self.llm = SimpleLLM(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.LLM_MODEL
        )
        
        logger.info("LLM initialized for agents")
    
    def build_query_techniques_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary of all 8 QueryTechnique instances.
        
        Returns:
            Dict mapping technique names to QueryTechnique instances
        """
        
        technique_names = [
            'standard',
            'multi_query',
            'rag_fusion',
            'step_back',
            'hyde',
            'query_decomposition',
            'query_rewriting',
            'query_expansion'
        ]
        
        techniques_dict = {}
        
        for technique_name in technique_names:
            try:
                technique = get_query_technique(
                    technique_name,
                    self.embedder,
                    self.hybrid_retriever or self.retriever,
                    self.generator,
                    self.config.QUERY_TECHNIQUE_CONFIG
                )
                techniques_dict[technique_name] = technique
                logger.info(f"  ✓ Loaded {technique_name}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {technique_name}: {e}")
        
        logger.info(f"Query techniques loaded: {list(techniques_dict.keys())}")
        return techniques_dict
    
    def build_agentic_graph(self):
        """Build the agentic graph with all agents."""
        
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Call initialize_components() first.")
        
        logger.info("Building agentic graph...")
        
        # Build query techniques dict
        query_techniques_dict = self.build_query_techniques_dict()
        
        if not query_techniques_dict:
            raise RuntimeError("Failed to load query techniques")
        
        # Build the graph
        config_dict = {
            'TOP_K': self.config.TOP_K,
            'MAX_RETRIES': getattr(self.config, 'MAX_RETRIES', 2)
        }
        
        self.agentic_graph = build_agentic_graph(
            self.llm,
            self.embedder,
            self.hybrid_retriever or self.retriever,
            self.generator,
            query_techniques_dict,
            config_dict
        )
        
        logger.info("Agentic graph built successfully")
    
    def run_query(self, question: str) -> Dict[str, Any]:
        """
        Run a single question through the agentic pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with question, agent_decisions, retrieved_docs, and answer
        """
        
        if self.agentic_graph is None:
            raise RuntimeError("Graph not built. Call build_agentic_graph() first.")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Running agentic pipeline for question: {question}")
        logger.info(f"{'='*80}")
        
        # Create initial state
        initial_state = AgenticRAGState(
            messages=[],  # Empty messages (we don't use chat history in this version)
            original_question=question,
            rewritten_queries=None,
            retrieved_documents=None,
            retrieved_text="",
            grade_decision="",
            grade_score="",
            grade_confidence=0.0,
            grade_reasoning="",
            retry_count=0,
            max_retries=getattr(self.config, 'MAX_RETRIES', 2),
            last_technique_used="",
            chosen_prompting_strategy="",
            generated_answer="",
            generation_confidence=0.0,
            agent_decisions={}
        )
        
        # Run the graph synchronously
        try:
            final_state = self.agentic_graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Error running graph: {e}")
            raise
        
        # Extract results (handle both dict and AgenticRAGState)
        if isinstance(final_state, dict):
            # retrieved_documents from agentic nodes are in Qdrant format with payload
            raw_docs = final_state.get('retrieved_documents') or []
            result = {
                "question": question,
                "answer": final_state.get('generated_answer', ''),
                "agent_decisions": final_state.get('agent_decisions') or {},
                "retrieved_documents": raw_docs,  # Keep raw Qdrant format for evaluation
                "retrieved_documents_raw": raw_docs,  # Explicitly mark for retrieval metrics
                "confidence": final_state.get('generation_confidence', 0.0),
                "num_docs_retrieved": len(raw_docs),
            }
        else:
            raw_docs = final_state.retrieved_documents or []
            result = {
                "question": question,
                "answer": final_state.generated_answer,
                "agent_decisions": final_state.agent_decisions or {},
                "retrieved_documents": raw_docs,  # Keep raw Qdrant format for evaluation
                "retrieved_documents_raw": raw_docs,  # Explicitly mark for retrieval metrics
                "confidence": final_state.generation_confidence,
                "num_docs_retrieved": len(raw_docs),
            }
        
        logger.info(f"\n{'='*80}")
        logger.info("FINAL ANSWER")
        logger.info(f"{'='*80}")
        logger.info(result["answer"][:500] + ("..." if len(result["answer"]) > 500 else ""))
        
        return result
    
    def evaluate(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the agentic pipeline on test questions.
        
        Args:
            test_questions: List of test question dicts with 'question' and 'answer' keys
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        if self.agentic_graph is None:
            raise RuntimeError("Graph not built. Call build_agentic_graph() first.")
        
        eval_start = time.time()
        logger.info(f"\nEvaluating agentic pipeline on {len(test_questions)} questions...")
        
        test_subset = test_questions[:self.config.EVAL_SUBSET_SIZE]
        
        results = []
        
        for i, test_q in enumerate(test_subset):
            logger.info(f"\n[{i+1}/{len(test_subset)}] {test_q['question'][:100]}...")
            
            try:
                result = self.run_query(test_q['question'])
                result['ground_truth'] = test_q.get('answer', '')
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                results.append({
                    "question": test_q['question'],
                    "answer": "",
                    "error": str(e),
                    "ground_truth": test_q.get('answer', '')
                })
        
        # Compute evaluation metrics (same as other systems for comparison)
        from evaluation.retrieval_metrics import evaluate_retrieval
        from evaluation.generation_metrics import evaluate_generation
        
        # Prepare docs for retrieval evaluation
        # evaluate_retrieval expects: List of retrieved document lists (one per query)
        # Each document should be in Qdrant format with id, score, text, payload (metadata)
        retrieval_results = [
            r.get("retrieved_documents_raw", [])  # Use raw Qdrant documents
            for r in results if "retrieved_documents_raw" in r
        ]
        
        # If raw documents not available, skip retrieval evaluation
        if not retrieval_results or len(retrieval_results) == 0:
            logger.warning("No retrieved documents in raw format for evaluation. Skipping retrieval metrics.")
            retrieval_metrics = {k: float('nan') for k in [
                'precision@1', 'precision@3', 'precision@5',
                'recall@1', 'recall@3', 'recall@5',
                'page_recall@1', 'page_recall@3', 'page_recall@5',
                'ndcg@1', 'ndcg@3', 'ndcg@5',
                'map', 'mrr'
            ]}
        else:
            # Evaluate
            retrieval_metrics = evaluate_retrieval(
                retrieval_results,
                test_subset,
                k_values=[1, 3, 5]
            )
        
        # Extract predictions and ground truths for generation evaluation
        predictions = [r.get("answer", "") for r in results]
        ground_truths = [r.get("ground_truth", "") for r in results]
        generation_metrics = evaluate_generation(predictions, ground_truths)
        
        elapsed = time.time() - eval_start
        
        eval_summary = {
            "num_questions": len(test_subset),
            "time_elapsed": elapsed,
            "time_per_question": elapsed / len(test_subset) if test_subset else 0,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "results": results
        }
        
        logger.info(f"\nEvaluation completed in {elapsed:.2f}s")
        
        return eval_summary
