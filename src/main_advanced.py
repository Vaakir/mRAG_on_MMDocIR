# src/main_advanced.py
# Advanced RAG pipeline entry point with query technique support

import os
import sys
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))

from config.config import AdvancedConfig
from data.data_loader import load_train_data
from pipelines.advanced_pipeline import AdvancedRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_query_test(pipeline: AdvancedRAGPipeline, question: str, ground_truth: Optional[str] = None):
    """Run and display results for a single query."""
    print("\n", "="*80, "SINGLE QUERY TEST", "="*80)
    
    result = pipeline.run_query(question, use_technique=True)
    
    print(f"\nQuestion: {question}")
    if ground_truth:
        print(f"Ground Truth: {ground_truth}")
    print(f"\nRetrieved {result['num_docs']} documents:")
    for i, doc in enumerate(result['retrieved_docs'][:3], 1):
        text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
        print(f"  [{i}] {text_preview}")
    
    print(f"\nGenerated Answer:\n{result['answer']}")


def main(technique: str = 'standard', prompting_strategy: str = 'standard', eval_subset: int = 20, force_rebuild: bool = False):
    """
    Main function to run the advanced pipeline.
    """
    metrics = {'timing': {}}

    @contextmanager
    def time_phase(name):
        start = time.time()
        yield
        metrics['timing'][name] = time.time() - start

    with time_phase('Total Pipeline Runtime'):
        logger.info(f"Starting ADVANCED RAG SYSTEM | Technique: {technique.upper()} | Prompting: {prompting_strategy.upper()}")
        
                #[standard, multi_query, rag_fusion, hyde, step_back,query_decomposition, query_rewriting, query_expansion]
        config = AdvancedConfig(
            QUERY_TECHNIQUE=technique,
            EVAL_SUBSET_SIZE = eval_subset,
            PROMPTING_STRATEGY = prompting_strategy
        )
        
        logger.info(f"Configuration -> Embed: {config.EMBEDDING_MODEL} | LLM: {config.LLM_MODEL} | DB: {config.VECTOR_DB_COLLECTION}")
        
        pipeline = AdvancedRAGPipeline(config)
        
        with time_phase('Index Build/Load'):
            pipeline.build_index(force_rebuild=force_rebuild)

        with time_phase('Component Initialization'):
            pipeline.initialize_components()

        with time_phase('Data Loading'):
            train_data = load_train_data(config.TRAIN_JSONL)
            logger.info(f"Loaded {len(train_data)} questions (all types)")

        if train_data:
            with time_phase('Single Query Test'):
                sample = train_data[0]
                run_single_query_test(pipeline, sample['question'], sample.get('answer'))

        with time_phase(f'Full Evaluation ({eval_subset} qs)'):
            eval_metrics = pipeline.evaluate(train_data[:eval_subset], use_technique=True)

        # Merge new metrics without clobbering inner timing dict
        metrics.update({k: v for k, v in eval_metrics.items() if k != 'timing'})
        if 'timing' in eval_metrics:
            metrics['timing'].update(eval_metrics['timing'])

    # Clean Printing
    print(f"\n{'='*50}\n              FINAL EVALUATION METRICS\n{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n--- {key.upper()} ---")
            for sub_key, sub_val in value.items():
                print(f"  {sub_key:<30}: {sub_val:.4f}" if isinstance(sub_val, float) else f"  {sub_key:<30}: {sub_val}")
        else:
            print(f"{key:<32}: {value:.4f}" if isinstance(value, float) else f"{key:<32}: {value}")
    print(f"\n{'='*50}\n")
    
    return metrics


def compare_techniques(techniques: list = None, eval_subset: int = 20):
    """
    Compare multiple query techniques with shared component initialization.
    
    Optimization: Build index and initialize generator ONCE, then swap technique configs.
    This saves significant time vs rebuilding for each technique.
    """
    if techniques is None:
        techniques = ['standard', 'multi_query', 'rag_fusion', 'step_back', 'hyde', 'query_decomposition', 'query_rewriting', 'query_expansion']
    
    logger.info(f"COMPARING QUERY TECHNIQUES: {', '.join(techniques)}")
    
    metrics = {'timing': {}}

    @contextmanager
    def time_phase(name):
        start = time.time()
        yield
        metrics['timing'][name] = time.time() - start

    with time_phase('Total Compare Runtime'):
        with time_phase('Shared Setup'):
            config = AdvancedConfig(EVAL_SUBSET_SIZE=eval_subset)
            pipeline = AdvancedRAGPipeline(config)
            
            logger.info("Building Index (Shared)...")
            pipeline.build_index(force_rebuild=False)
            
            logger.info("Initializing Components (Shared)...")
            pipeline.initialize_components()
            
            logger.info("Loading Test Data...")
            train_data = load_train_data(config.TRAIN_JSONL)

        results = {}
        for technique in techniques:
            logger.info(f"--- Evaluating Technique: {technique.upper()} ---")
            try:
                tech_config = AdvancedConfig(
                    QUERY_TECHNIQUE=technique,
                    EVAL_SUBSET_SIZE=eval_subset
                )

                pipeline.config = tech_config
                pipeline.initialize_components()

                tech_metrics = pipeline.evaluate(train_data[:eval_subset], use_technique=True)
                results[technique] = tech_metrics
                
                # Print quick snapshot
                if tech_metrics and 'retrieval' in tech_metrics:
                    ret = tech_metrics['retrieval']
                    logger.info(f"[{technique.upper()}] P@1: {ret.get('precision@1', 0):.4f} | R@5: {ret.get('recall@5', 0):.4f} | MRR: {ret.get('mrr', 0):.4f}")
            except Exception as e:
                logger.error(f"ERROR: {technique} failed - {e}")
                results[technique] = None

    # ===== SUMMARY COMPARISON =====
    print(f"\n\n{'='*90}\n                   COMPARISON SUMMARY\n{'='*90}")
    
    print(f"\n{'RETRIEVAL METRICS':<20} | {'P@1':<8} | {'P@3':<8} | {'P@5':<8} | {'R@1':<8} | {'R@3':<8} | {'R@5':<8} | {'MRR':<8}")
    print("-" * 90)
    for tech, met in results.items():
        if met and 'retrieval' in met:
            r = met['retrieval']
            print(f"{tech:<20} | {r.get('precision@1',0):<8.4f} | {r.get('precision@3',0):<8.4f} | {r.get('precision@5',0):<8.4f} | {r.get('recall@1',0):<8.4f} | {r.get('recall@3',0):<8.4f} | {r.get('recall@5',0):<8.4f} | {r.get('mrr',0):<8.4f}")
    
    print(f"\n{'GENERATION METRICS':<20} | {'Token F1':<12} | {'BLEU':<12} | {'ROUGE-L':<12} | {'Exact Match':<12}")
    print("-" * 80)
    for tech, met in results.items():
        if met and 'generation' in met:
            g = met['generation']
            print(f"{tech:<20} | {g.get('token_f1',0):<12.4f} | {g.get('bleu',0):<12.4f} | {g.get('rouge_l',0):<12.4f} | {g.get('exact_match',0):<12.4f}")
    
    total_eval_time = sum([r['timing']['total'] for r in results.values() if r and 'timing' in r], 0)
    valid_evals = len([r for r in results.values() if r and 'timing' in r])
    
    print(f"\n{'TIMING SUMMARY':<20} | SETUP: {metrics['timing'].get('Shared Setup', 0):.2f}s | TOTAL (ALL EVALS): {metrics['timing'].get('Total Compare Runtime', 0):.2f}s | AVG EVAL: {(total_eval_time/max(valid_evals, 1)):.2f}s")
    print(f"{'='*90}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline with Query Techniques and Prompting Strategies")
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
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple techniques'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_techniques(eval_subset=args.eval_subset)
    else:
        main(technique=args.technique, prompting_strategy=args.prompting_strategy, eval_subset=args.eval_subset, force_rebuild=args.force_rebuild)
