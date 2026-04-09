# src/main_advanced.py
# Advanced RAG pipeline entry point with query technique support

import os
import sys
from pathlib import Path
from typing import Optional

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))

from config.config import AdvancedConfig, MultiQueryConfig, RAGFusionConfig, HyDEConfig
from data.data_loader import load_train_data
from pipelines.advanced_pipeline import AdvancedRAGPipeline


def get_config(technique: str = 'standard') -> AdvancedConfig:
    """
    Get configuration for a specific query technique.
    
    Args:
        technique: Query technique name or 'standard'
        
    Returns:
        AdvancedConfig instance with technique set
    """
    technique = technique.lower().strip()
    
    # Create base config
    if technique == 'multi_query':
        config = MultiQueryConfig()
    elif technique == 'rag_fusion':
        config = RAGFusionConfig()
    elif technique == 'hyde':
        config = HyDEConfig()
    else:
        config = AdvancedConfig()
    
    # Explicitly set technique to ensure it's applied
    # (dataclass inheritance doesn't override defaults reliably)
    if technique == 'step_back':
        config.QUERY_TECHNIQUE = 'step_back'
    elif technique == 'query_decomposition':
        config.QUERY_TECHNIQUE = 'query_decomposition'
    elif technique == 'query_rewriting':
        config.QUERY_TECHNIQUE = 'query_rewriting'
    elif technique == 'query_expansion':
        config.QUERY_TECHNIQUE = 'query_expansion'
    elif technique != 'standard':
        # For preset configs (multi_query, rag_fusion, hyde), ensure technique is set
        config.QUERY_TECHNIQUE = technique
    
    return config


def run_single_query_test(pipeline: AdvancedRAGPipeline, question: str, ground_truth: Optional[str] = None):
    """Run and display results for a single query."""
    print("\n" + "="*80)
    print("SINGLE QUERY TEST")
    print("="*80)
    
    result = pipeline.run_query(question, use_technique=True)
    
    print(f"\nQuestion: {question}")
    if ground_truth:
        print(f"Ground Truth: {ground_truth}")
    print(f"\nRetrieved {result['num_docs']} documents:")
    for i, doc in enumerate(result['retrieved_docs'][:3], 1):
        text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
        print(f"  [{i}] {text_preview}")
    
    print(f"\nGenerated Answer:\n{result['answer']}")


def main(technique: str = 'standard', eval_subset: int = 20, force_rebuild: bool = False):
    """
    Main function to run the advanced pipeline.
    
    Args:
        technique: Query technique to use ('standard', 'multi_query', 'rag_fusion', 'hyde', etc.)
        eval_subset: Number of test questions to evaluate on
        force_rebuild: Force rebuild of the index
    """
    import time
    main_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"ADVANCED RAG PIPELINE - Query Technique: {technique.upper()}")
    print(f"{'='*80}\n")
    
    # Get configuration
    config = get_config(technique)
    config.EVAL_SUBSET_SIZE = eval_subset
    
    print(f"Configuration:")
    print(f"  Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"  LLM Model: {config.LLM_MODEL}")
    print(f"  Vector DB: {config.VECTOR_DB_COLLECTION}")
    print(f"  Query Technique: {config.QUERY_TECHNIQUE}")
    print(f"  Retrieved Top-K: {config.TOP_K}")
    if 'num_variants' in config.QUERY_TECHNIQUE_CONFIG:
        print(f"  Technique Variants: {config.QUERY_TECHNIQUE_CONFIG['num_variants']}")
    
    # Initialize pipeline
    pipeline = AdvancedRAGPipeline(config)
    
    # Build/load index
    print(f"\n{'='*80}")
    print("BUILDING INDEX")
    print(f"{'='*80}")
    build_start = time.time()
    pipeline.build_index(force_rebuild=force_rebuild)
    build_time = time.time() - build_start
    print(f"\nIndex build completed in {build_time:.2f}s")
    
    # Initialize components (generator + query technique)
    print(f"\n{'='*80}")
    print("INITIALIZING COMPONENTS")
    print(f"{'='*80}")
    init_start = time.time()
    pipeline.initialize_components()
    init_time = time.time() - init_start
    print(f"\nComponents initialized in {init_time:.2f}s")
    
    # Load test data
    print(f"\nLoading test data...")
    train_data = load_train_data(config.TRAIN_JSONL)
    print(f"Loaded {len(train_data)} questions (all types)")

    # Test single query
    if train_data:
        sample = train_data[0]
        run_single_query_test(
            pipeline,
            sample['question'],
            sample.get('answer')
        )

    # Run evaluation
    print(f"\n{'='*80}")
    print(f"RUNNING EVALUATION (All Question Types)")
    print(f"{'='*80}")
    metrics = pipeline.evaluate(train_data[:eval_subset], use_technique=True)
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\nRetrieval Metrics:")
    for key, value in metrics['retrieval'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nGeneration Metrics:")
    for key, value in metrics['generation'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Print timing information
    if 'timing' in metrics:
        print(f"\nTiming:")
        print(f"  Index Build: {build_time:.2f}s")
        print(f"  Component Init: {init_time:.2f}s")
        print(f"  Phase 1 (Retrieval): {metrics['timing']['phase1']:.2f}s")
        print(f"  Phase 2 (Generation): {metrics['timing']['phase2']:.2f}s")
        print(f"  Total Evaluation: {metrics['timing']['total']:.2f}s")
    
    main_total = time.time() - main_start
    print(f"\n  Total Runtime: {main_total:.2f}s")
    
    return metrics


def compare_techniques(techniques: list = None, eval_subset: int = 20):
    """
    Compare multiple query techniques with shared component initialization.
    
    Optimization: Build index and initialize generator ONCE, then swap technique configs.
    This saves significant time vs rebuilding for each technique.
    
    Args:
        techniques: List of technique names to compare
        eval_subset: Number of test questions per technique
    """
    import time
    compare_start = time.time()
    
    if techniques is None:
        techniques = ['standard', 'multi_query', 'rag_fusion', 'step_back', 'hyde', 'query_decomposition', 'query_rewriting', 'query_expansion']
    
    print(f"\n{'='*80}")
    print("COMPARING QUERY TECHNIQUES")
    print(f"{'='*80}\n")
    
    # ===== SHARED SETUP (done once) =====
    print("SHARED SETUP (done once for all techniques):")
    setup_start = time.time()
    
    # Build index once with a base config
    config = AdvancedConfig()
    config.EVAL_SUBSET_SIZE = eval_subset
    
    pipeline = AdvancedRAGPipeline(config)
    
    print(f"\nBuilding index...")
    build_start = time.time()
    pipeline.build_index(force_rebuild=False)
    build_time = time.time() - build_start
    print(f"  [OK] Index built in {build_time:.2f}s")
    
    print(f"\nInitializing components (generator + VLM + query technique)...")
    gen_start = time.time()
    pipeline.initialize_components()
    gen_time = time.time() - gen_start
    print(f"  [OK] Components initialized in {gen_time:.2f}s")

    # Load test data once
    print(f"\nLoading test data...")
    train_data = load_train_data(config.TRAIN_JSONL)
    print(f"  [OK] Loaded {len(train_data)} questions (all types)")
    
    setup_time = time.time() - setup_start
    print(f"\nShared setup completed in {setup_time:.2f}s\n")
    
    # ===== TECHNIQUE EVALUATION (swaps technique config only) =====
    results = {}
    
    for technique in techniques:
        print(f"\n{'='*80}")
        print(f"Evaluating technique: {technique.upper()}")
        print(f"{'='*80}")
        
        try:
            # Create technique-specific config
            tech_config = get_config(technique)
            tech_config.EVAL_SUBSET_SIZE = eval_subset
            
            # Update pipeline config and reinitialize ONLY the query technique
            pipeline.config = tech_config
            pipeline.initialize_components()
            
            # Run evaluation
            metrics = pipeline.evaluate(train_data, use_technique=True)
            results[technique] = metrics
            
            # Print quick results for this technique
            if metrics and 'retrieval' in metrics:
                ret = metrics['retrieval']
                print(f"\n  Retrieval - P@1: {ret.get('precision@1', 0):.4f}, R@5: {ret.get('recall@5', 0):.4f}, MRR: {ret.get('mrr', 0):.4f}")
            if metrics and 'generation' in metrics:
                gen = metrics['generation']
                print(f"  Generation - Token F1: {gen.get('token_f1', 0):.4f}, BLEU: {gen.get('bleu', 0):.4f}")
            if metrics and 'timing' in metrics:
                print(f"  Timing - Total: {metrics['timing']['total']:.2f}s (P1: {metrics['timing']['phase1']:.2f}s, P2: {metrics['timing']['phase2']:.2f}s)")
        
        except Exception as e:
            print(f"ERROR: {technique} failed - {e}")
            results[technique] = None
    
    # ===== SUMMARY COMPARISON =====
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print("RETRIEVAL METRICS:")
    print(f"{'Technique':<20} {'P@1':<10} {'P@3':<10} {'P@5':<10} {'R@1':<10} {'R@3':<10} {'R@5':<10} {'MRR':<10}")
    print("-" * 90)
    
    for technique, metrics in results.items():
        if metrics and 'retrieval' in metrics:
            ret = metrics['retrieval']
            p1 = ret.get('precision@1', 0)
            p3 = ret.get('precision@3', 0)
            p5 = ret.get('precision@5', 0)
            r1 = ret.get('recall@1', 0)
            r3 = ret.get('recall@3', 0)
            r5 = ret.get('recall@5', 0)
            mrr = ret.get('mrr', 0)
            print(f"{technique:<20} {p1:<10.4f} {p3:<10.4f} {p5:<10.4f} {r1:<10.4f} {r3:<10.4f} {r5:<10.4f} {mrr:<10.4f}")
    
    print(f"\nGENERATION METRICS:")
    print(f"{'Technique':<20} {'Token F1':<15} {'BLEU':<15} {'ROUGE-L':<15} {'Exact Match':<15}")
    print("-" * 80)
    
    for technique, metrics in results.items():
        if metrics and 'generation' in metrics:
            gen = metrics['generation']
            f1 = gen.get('token_f1', 0)
            bleu = gen.get('bleu', 0)
            rouge = gen.get('rouge_l', 0)
            exact = gen.get('exact_match', 0)
            print(f"{technique:<20} {f1:<15.4f} {bleu:<15.4f} {rouge:<15.4f} {exact:<15.4f}")
    
    # Print timing summary
    compare_total = time.time() - compare_start
    total_eval_time = sum([r['timing']['total'] for r in results.values() if r and 'timing' in r], 0)
    avg_eval_time = total_eval_time / len([r for r in results.values() if r and 'timing' in r])
    
    print(f"\n\nTIMING SUMMARY:")
    print(f"{'='*80}")
    print(f"  Shared setup (index + generator + data): {setup_time:.2f}s")
    print(f"  Average per technique evaluation: {avg_eval_time:.2f}s")
    print(f"  Total evaluations ({len(techniques)} techniques): {total_eval_time:.2f}s")
    print(f"  Total runtime: {compare_total:.2f}s")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline with Query Techniques")
    parser.add_argument(
        '--technique',
        type=str,
        default='standard',
        help='Query technique to use (standard, multi_query, rag_fusion, step_back, hyde, query_decomposition, query_rewriting, query_expansion)'
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
        main(technique=args.technique, eval_subset=args.eval_subset, force_rebuild=args.force_rebuild)
