# src/main_baseline.py

import os
import sys
import time
import logging
import pandas as pd
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import BaselineConfig
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    timings = {}
    main_start = time.time()
    
    logger.info("Starting BASELINE RAG SYSTEM (System 1)")
    
    config = BaselineConfig()
    pipeline = BaselineRAGPipeline(config)
    
    # Phase 1: Index Building/Loading
    build_start = time.time()
    pipeline.build_index(force_rebuild=False)
    timings['Index Build/Load'] = time.time() - build_start

    # Phase 2: Component Initialization
    init_start = time.time()
    pipeline.initialize_components()
    timings['Component Initialization'] = time.time() - init_start

    # Phase 3: Data Loading and Filtering
    data_start = time.time()
    train_data = load_train_data(config.TRAIN_JSONL)
    pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]]
    timings['Data Loading/Filtering'] = time.time() - data_start

    logger.info(f"Loaded {len(train_data)} total questions, filtered to {len(pure_text_data)} pure-text questions")

    # Phase 4: Single Query Test
    sample_query = pure_text_data[0]
    query_start = time.time()
    result = pipeline.run_query(sample_query["question"])
    timings['Single Query Test'] = time.time() - query_start
    
    logger.info(f"Single query completed. Question: {sample_query['question']} | Ground Truth: {sample_query['answer']} | Generated: {result['answer']}")

    # Phase 5: Full Evaluation
    eval_start = time.time()
    metrics = pipeline.evaluate(pure_text_data[:20])
    eval_time = time.time() - eval_start
    timings['Full Evaluation (20 qs)'] = eval_time

    main_total = time.time() - main_start
    timings['Total Runtime'] = main_total
    
    print(f"[OK] Evaluation completed in {eval_time:.2f}s\n")
    
    # ===== SUMMARY REPORT =====
    print(f"\n{'='*80}")
    print("BASELINE SYSTEM - TIMING SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Phase Breakdown:")
    print(f"  1. Index Build/Load:               {timings['Index Build/Load']:>10.2f}s")
    print(f"  2. Component Initialization:       {timings['Component Initialization']:>10.2f}s")
    print(f"  3. Data Loading/Filtering:         {timings['Data Loading/Filtering']:>10.2f}s")
    print(f"  4. Single Query Test:              {timings['Single Query Test']:>10.2f}s")
    print(f"  5. Full Evaluation (20 qs):        {timings['Full Evaluation (20 qs)']:>10.2f}s")
    print(f"  " + "-" * 50)
    
    print(f"  TOTAL RUNTIME:                     {timings['Total Runtime']:>10.2f}s\n")
    
    # Print metrics if available
    if metrics:
        print(f"\n{'='*80}")
        print("EVALUATION METRICS")
        print(f"{'='*80}\n")
        
        if 'retrieval' in metrics:
            print("Retrieval Metrics:")
            for key, value in metrics['retrieval'].items():
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:.4f}")
                else:
                    print(f"  {key:<20}: {value}")
        
        if 'generation' in metrics:
            print("\nGeneration Metrics:")
            for key, value in metrics['generation'].items():
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:.4f}")
                else:
                    print(f"  {key:<20}: {value}")
        
        # Print phase-specific timing if available in metrics
        if 'timing' in metrics:
            print("\nPhase-Specific Timing (Evaluation):")
            print(f"  Retrieval Phase:   {metrics['timing'].get('phase1', 0):.2f}s")
            print(f"  Generation Phase:  {metrics['timing'].get('phase2', 0):.2f}s")
            print(f"  Total Evaluation:  {metrics['timing'].get('total', 0):.2f}s")
    
    print(f"\n{'='*80}\n")
    
    return metrics

if __name__ == "__main__":
    main()