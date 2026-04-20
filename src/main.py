# src/main_advanced.py
# Advanced RAG pipeline entry point with query technique support

import os
import sys
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['USE_TF'] = '0'

sys.path.append(str(Path(__file__).parent))

from config.config import AdvancedConfig, BaselineConfig, PIPELINE_TIME_CSV, RESULTS_CSV
from utils.timer import MetricsTracker
from data.data_loader import load_train_data
from pipelines.advanced_pipeline import AdvancedRAGPipeline
from pipelines.baseline_pipeline import BaselineRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_query_test(pipeline, question: str, ground_truth: Optional[str] = None):
    """Run and display results for a single query."""
    logger.info(f"SINGLE QUERY TEST -> '{question}'")
    
    result = pipeline.run_query(question, use_technique=True)
    
    num_docs = result.get('num_docs', len(result.get('retrieved_docs', [])))
    logger.info(f"Retrieved {num_docs} documents.")
    
    for i, doc in enumerate(result.get('retrieved_docs', [])[:3], 1):
        text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
        logger.info(f"  [{i}] {text_preview}")
    
    answer = result.get('answer', '')
    logger.info(f"Generated Answer: {answer[:100]}...")


def run_single_experiment(
    experiment_name: str, 
    config, 
    pipeline_class, 
    test_data: List[Dict[str, Any]],
    force_rebuild: bool = False,
    run_single_query: bool = False,
    results_csv_path: Optional[Path] = RESULTS_CSV,
    time_csv_path: Optional[Path] = PIPELINE_TIME_CSV,
):
    """Run a single pipeline configuration and log metrics."""
    logger.info(f"\n{'='*80}\nSTARTING EXPERIMENT: {experiment_name}\n{'='*80}")
    
    full_pipeline_name = f"{experiment_name} ({pipeline_class.__name__})"
    tracker = MetricsTracker(logger, pipeline_name=full_pipeline_name, model_name=config.LLM_MODEL)
    log_and_time = tracker.log_and_time
    
    pipeline = pipeline_class(config)
    
    with log_and_time('Total Pipeline Runtime'):

        with log_and_time('Index Build/Load'):
            pipeline.build_index(force_rebuild=force_rebuild)
            
        with log_and_time('Component Initialization'):
            pipeline.initialize_components()
            
        if run_single_query and not test_data:
            raise ValueError("no test_data found")
        
        with log_and_time('Single Query Test'):
            run_single_query_test(pipeline, test_data[0]['question'], test_data[0].get('answer'))
        
        # Run evaluation
        with log_and_time('Full Evaluation Time'):
            metrics = pipeline.evaluate(test_data, use_technique=True, experiment_name=experiment_name)
    
    tracker.save_to_csv(metrics, results_csv_path)
    tracker.save_to_csv(tracker.timing_data, time_csv_path)
    tracker.print_metrics(f"RESULTS FOR: {experiment_name}", metrics)

    return metrics

def run_experiments(eval_subset_size: int = 50):
    """
    Run full ablations comparing baseline and advanced techniques.
    """
    logger.info("Loading evaluation data...")
    
    # EXPERIMENT 1: BASELINE
    baseline_cfg = BaselineConfig(EVAL_SUBSET_SIZE=eval_subset_size)
    train_data = load_train_data(baseline_cfg.TRAIN_JSONL)
    run_single_experiment("1_Baseline", baseline_cfg, BaselineRAGPipeline, train_data)

    # Helper function to run ablation variations cleanly
    def _run_ablation(exp_id: str, exp_desc: str, param_key: str, param_val: Any, **extra_kwargs):
        opts = {
            "EVAL_SUBSET_SIZE": eval_subset_size,
            "CHUNKING_STRATEGY": "fixed_size",
            "QUERY_TECHNIQUE": "standard",
            "PROMPTING_STRATEGY": "standard"
        }
        opts[param_key] = param_val
        opts.update(extra_kwargs)
        
        chunk_strat = opts["CHUNKING_STRATEGY"]
        opts["PREPROCESSED_CHUNKS_FILE"] = str(Path(baseline_cfg.SRC_DIR) / "data" / "preprocessed" / f"chunks_{chunk_strat}.json")
        opts["VECTOR_DB_COLLECTION"] = opts.get("VECTOR_DB_COLLECTION", f"bge_collection_{chunk_strat}")
        
        cfg = AdvancedConfig(**opts)
        exp_name = f"{exp_id}_{exp_desc}_{param_val}"
        
        try:
            run_single_experiment(exp_name, cfg, AdvancedRAGPipeline, train_data)
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")

    for chuking in ["fixed_size", "sliding_window", "semantic", "hierarchical"]:
        _run_ablation("2", "Chunk_Ablation", "CHUNKING_STRATEGY", chuking)

    for query_processing in ["standard", "multi_query", "rag_fusion", "step_back", "hyde", "query_decomposition", "query_rewriting", "query_expansion"]:
        _run_ablation("3", "Query_Ablation", "QUERY_TECHNIQUE", query_processing)

    for prompting_strategy in ["standard", "few_shot", "role", "cot"]:
        _run_ablation("4", "Prompt_Ablation", "PROMPTING_STRATEGY", prompting_strategy)

    _run_ablation("5", "Multimodal", "USE_MULTIMODAL", True, 
                  EMBEDDING_MODEL="jinaai/jina-clip-v2", 
                  VECTOR_DB_COLLECTION="baseline_documents_jina")

    logger.info("\nAll experiments completed! Check experiments_results.csv for details.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline Runner")
    parser.add_argument(
        '--run-experiments',
        action='store_true',
        help='Run all ablation experiments instead of a single query test.'
    )
    parser.add_argument(
        '--technique',
        type=str,
        default='standard',
        help='Query technique to use (standard, multi_query, rag_fusion, step_back, hyde, query_decomposition, query_rewriting, query_expansion)'
    )
    parser.add_argument(
        '--prompting-strategy',
        type=str,
        default='standard',
        help='Prompting strategy for answer generation (standard, role, cot, ensemble)'
    )
    parser.add_argument(
        '--eval-subset',
        type=int,
        default=20,
        help='Number of test questions to evaluate on'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of the index'
    )
    
    args = parser.parse_args()
    
    if args.run_experiments:
        config = AdvancedConfig(
            QUERY_TECHNIQUE=args.technique,
            prompting_strategy=args.prompting_strategy,
            EVAL_SUBSET_SIZE=args.eval_subset,
        )
        train_data = load_train_data(config.TRAIN_JSONL)        
        run_single_experiment(
            experiment_name=f"Advanced RAG ({config.QUERY_TECHNIQUE})",
            config=config,
            pipeline_class=AdvancedRAGPipeline,
            test_data=train_data[:config.EVAL_SUBSET_SIZE],
            force_rebuild=args.force_rebuild,
            run_single_query=True
        )
    else:
        # How to use - DAT560project> python src/main.py:
         
        # # TO RUN BASELINE:
        # config = BaselineConfig()
        # pure_text_data = load_train_data(config.TRAIN_JSONL)
        # run_single_experiment(
        #     experiment_name="Baseline RAG",
        #     config=config,
        #     pipeline_class=BaselineRAGPipeline,
        #     test_data=pure_text_data[:config.EVAL_SUBSET_SIZE],
        #     force_rebuild=False,
        #     run_single_query=True
        # )

        # # TO RUN ALL ABLATION TESTS - DAT560project> python src/main.py:
        run_experiments(eval_subset_size=2)
