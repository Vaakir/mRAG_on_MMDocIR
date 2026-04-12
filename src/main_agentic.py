"""
System 3: Agentic mRAG Pipeline Main Entry Point

This script runs the full agentic RAG pipeline with agent-based decision making:
- Query Rewriter Agent: Decides which query technique to use
- Retriever: Uses the chosen technique to retrieve documents
- Grader Agent: Evaluates document relevance and decides if retry is needed
- Generator Agent: Decides prompting strategy and generates answer

Usage:
    python src/main_agentic.py --test-query "What is X?" --num-queries 5
    python src/main_agentic.py --eval --eval-size 10
"""

import sys
import os
import logging
import json
import time
from pathlib import Path
from typing import Optional

import argparse

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import AdvancedConfig
from data.data_loader import load_train_data
from pipelines.agentic_pipeline import AgenticRAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for System 3 agentic pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Run System 3 Agentic mRAG Pipeline"
    )
    
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Test with a single query"
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run full evaluation"
    )
    
    parser.add_argument(
        "--eval-size",
        type=int,
        default=5,
        help="Number of test queries to evaluate (default: 5)"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of index"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SYSTEM 3: AGENTIC mRAG PIPELINE")
    logger.info("="*80)
    
    # Initialize config
    logger.info("Loading configuration...")
    config = AdvancedConfig()
    config.EVAL_SUBSET_SIZE = args.eval_size
    
    # Initialize pipeline
    logger.info("Initializing agentic pipeline...")
    pipeline = AgenticRAGPipeline(config)
    
    # Build index
    logger.info("Building/loading index...")
    start = time.time()
    pipeline.build_index(force_rebuild=args.rebuild_index)
    index_time = time.time() - start
    logger.info(f"Index ready in {index_time:.2f}s")
    
    # Initialize components
    logger.info("Initializing components...")
    pipeline.initialize_components()
    
    # Build agentic graph
    logger.info("Building agentic graph...")
    pipeline.build_agentic_graph()
    
    results = None
    
    # Test with single query
    if args.test_query:
        logger.info(f"\nTesting with query: {args.test_query}")
        result = pipeline.run_query(args.test_query)
        results = {"single_query": result}
        
        # Print result
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nAgent Decisions: {json.dumps(result['agent_decisions'], indent=2)}")
        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"Docs Retrieved: {result['num_docs_retrieved']}")
    
    # Full evaluation
    elif args.eval:
        logger.info(f"\nRunning evaluation on {args.eval_size} queries...")
        
        # Load test data
        test_data = load_train_data(config.TEST_JSONL)
        
        # Run evaluation
        eval_summary = pipeline.evaluate(test_data)
        
        results = eval_summary
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Questions evaluated: {eval_summary['num_questions']}")
        print(f"Total time: {eval_summary['time_elapsed']:.2f}s")
        print(f"Time per question: {eval_summary['time_per_question']:.2f}s")
        
        print("\nRetrieval Metrics:")
        for k, metrics in eval_summary['retrieval_metrics'].items():
            print(f"  {k}: {metrics}")
        
        print("\nGeneration Metrics:")
        for metric, value in eval_summary['generation_metrics'].items():
            print(f"  {metric}: {value}")
    
    else:
        # Default: test with sample query
        logger.info("\nNo query or eval specified. Running sample query...")
        sample_query = "What is the main topic of the documents?"
        result = pipeline.run_query(sample_query)
        results = {"sample_query": result}
        
        print("\n" + "="*80)
        print("SAMPLE RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']:.2f}")
    
    # Save results if requested
    if args.output and results:
        logger.info(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            # Convert any non-serializable objects
            def default_handler(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            json.dump(
                results,
                f,
                indent=2,
                default=default_handler
            )
    
    logger.info("\nPipeline completed!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
