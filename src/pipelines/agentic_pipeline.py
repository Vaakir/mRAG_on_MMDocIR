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
from agentic.llm import SimpleLLM
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

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
        self.agentic_graph = None # Will hold the compiled LangGraph StateGraph
        self.llm = None           # LLM for agent decision-making
        
    def initialize_components(self):
        """Initialize all components including separate LLMs for agent decisions and generation."""
        super().initialize_components()
        
        # Initialize lightweight LLM for agent decision-making (Query Rewriter, Grader, Generator strategy)
        print(f"Initializing lightweight LLM for agent decisions: {self.config.AGENT_LLM_MODEL}")
        
        self.agent_llm = SimpleLLM(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.AGENT_LLM_MODEL
        )
        
        print(f"Agent LLM initialized: {self.config.AGENT_LLM_MODEL}")
        print(f"Generator LLM (inherited from parent): {self.config.LLM_MODEL}")
    
    def build_query_techniques_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary of all 8 QueryTechnique instances.
        
        Returns:
            Dict mapping technique names to QueryTechnique instances
        """
        # Build the query techniques dict by instantiating each technique with required dependencies
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
        
        # Instantiate each technique and add to the dict
        for technique_name in technique_names:
            try:
                technique = get_query_technique( # Factory function to get the appropriate technique instance
                    technique_name,
                    self.embedder,
                    self.hybrid_retriever or self.retriever,
                    self.generator,
                    self.config.QUERY_TECHNIQUE_CONFIG
                )
                techniques_dict[technique_name] = technique # Store the instance in the dict
                print(f"  [OK]Loaded {technique_name}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {technique_name}: {e}")
        
        print(f"Query techniques loaded: {list(techniques_dict.keys())}")
        return techniques_dict
    
    def build_agentic_graph(self):
        """Build the agentic graph with all agents."""
        
        if self.agent_llm is None:
            raise RuntimeError("Agent LLM not initialized. Call initialize_components() first.")
        
        print("Building agentic graph...")
        
        # Build query techniques dict
        query_techniques_dict = self.build_query_techniques_dict()
        
        if not query_techniques_dict:
            raise RuntimeError("Failed to load query techniques")
        
        # Build the graph
        config_dict = {
            'TOP_K': self.config.TOP_K,
            'MAX_RETRIES': getattr(self.config, 'MAX_RETRIES', 1)
        }
        
        self.agentic_graph = build_agentic_graph( # Build the LangGraph StateGraph using the builder function, passing all dependencies
            self.agent_llm,  # Lightweight LLM for agent decisions
            self.embedder,
            self.hybrid_retriever or self.retriever,
            self.generator,  # Heavy LLM for final answer generation (self.generator uses qwen3:32b)
            query_techniques_dict,
            config_dict
        )
        
        print("Agentic graph built successfully")
    
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
        
        print(f"\n{'='*80}")
        print(f"Running agentic pipeline for question: {question}")
        print(f"{'='*80}")
        
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
            max_retries=getattr(self.config, 'MAX_RETRIES', 1), # Use config value or default to 2
            last_technique_used="",
            chosen_prompting_strategy="",
            generated_answer="",
            generation_confidence=0.0,
            agent_decisions={}
        )
        
        # Run the graph synchronously
        try:
            final_state = self.agentic_graph.invoke(initial_state) # Invoke the graph with the initial state and get the final state after all agents have processed
        except Exception as e:
            print(f"Error running graph: {e}")
            raise
        
        # Extract results (handle both dict and AgenticRAGState)
        if isinstance(final_state, dict): # If the graph returns a dict instead of an AgenticRAGState
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
        else: # Assume it's an AgenticRAGState and extract fields accordingly
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
        
        print("FINAL ANSWER\n" + result["answer"][:500] + ("..." if len(result["answer"]) > 500 else ""))
        
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
        print(f"\nEvaluating agentic pipeline on {len(test_questions)} questions...")
        
        test_subset = test_questions[:self.config.EVAL_SUBSET_SIZE] # Limit to subset for faster evaluation during development
        
        results = []
        
        # Run each test question through the pipeline and collect results
        for i, test_q in enumerate(test_subset):
            print(f"\n[{i+1}/{len(test_subset)}] Question: {test_q['question'][:100]}...")
            
            try:
                result = self.run_query(test_q['question']) # Run the question through the agentic pipeline and get the result
                result['ground_truth'] = test_q.get('answer', '') # Add ground truth to the result for later evaluation
                results.append(result)
            except Exception as e:
                print(f"Failed to process question: {e}")
                results.append({
                    "question": test_q['question'],
                    "answer": "",
                    "error": str(e),
                    "ground_truth": test_q.get('answer', '')
                })
        
        # Compute evaluation metrics (same as other systems for comparison)
        # Prepare docs for retrieval evaluation
        # evaluate_retrieval expects: List of retrieved document lists (one per query)
        # Each document should be in Qdrant format with id, score, text, payload (metadata)
        retrieval_results = [
            r.get("retrieved_documents_raw", [])  # Use raw Qdrant documents
            for r in results if "retrieved_documents_raw" in r
        ]
        
        # If raw documents not available, skip retrieval evaluation
        if not retrieval_results or len(retrieval_results) == 0:
            print("No retrieved documents in raw format for evaluation. Skipping retrieval metrics.")
            retrieval_metrics = {k: float('nan') for k in [
                'precision@1', 'precision@3', 'precision@5',
                'recall@1', 'recall@3', 'recall@5',
                'page_recall@1', 'page_recall@3', 'page_recall@5',
                'ndcg@1', 'ndcg@3', 'ndcg@5',
                'map', 'mrr'
            ]}
        else:
            # Evaluate retrieval using the raw retrieved documents and ground truth answers
            retrieval_metrics = evaluate_retrieval(
                retrieval_results, 
                test_subset,       
                k_values=[1, 3, 5]
            )
        
        # Extract predictions and ground truths for generation evaluation
        predictions = [r.get("answer", "") for r in results]
        ground_truths = [r.get("ground_truth", "") for r in results]
        generation_metrics = evaluate_generation( 
            predictions, 
            ground_truths,
            embedder=self.embedder  # Pass embedder for true semantic similarity computation
        )
        
        elapsed = time.time() - eval_start
        
        eval_summary = {
            "num_questions": len(test_subset),
            "time_elapsed": elapsed,
            "time_per_question": elapsed / len(test_subset) if test_subset else 0,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "results": results
        }
        
        print(f"\nEvaluation completed in {elapsed:.2f}s")
        
        return eval_summary
