# src/main_baseline.py

import os
import sys
import time
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import TRAIN_JSONL
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

def main():
    """Main function for baseline RAG pipeline with comprehensive timing instrumentation."""
    main_start = time.time() # Track total runtime of the main function for reference and monitoring purposes
    
    print(f"\n{'='*80}")
    print("BASELINE RAG SYSTEM (System 1)")
    print(f"{'='*80}\n")
    
    # Initialize pipeline
    pipeline = BaselineRAGPipeline()
    
    # ===== PHASE 1: Index Building/Loading =====
    print("===========PHASE 1: Building/Loading Index=============")
    build_start = time.time() # Track time for index building/loading phase for reference and monitoring purposes
    
    pipeline.build_index(force_rebuild=False) # 'True' if you want to rebuild the index from raw PDFs, 'False' to load existing index (must be built at least once)
    
    build_time = time.time() - build_start # Time taken for index building/loading phase for reference and monitoring purposes
    print(f"[OK] Index build/load completed in {build_time:.2f}s\n")

    # ===== PHASE 2: Component Initialization =====
    print("-" * 80)
    print("===========PHASE 2: Initializing Components=============")
    init_start = time.time() # Track time for component initialization phase for reference and monitoring purposes
    
    pipeline.initialize_components()
    
    init_time = time.time() - init_start # Time taken for component initialization phase for reference and monitoring purposes
    print(f"[OK] Components initialized in {init_time:.2f}s\n")

    # ===== PHASE 3: Data Loading and Filtering =====
    print("-" * 80)
    print("===========PHASE 3: Loading and Filtering Test Data=============")
    data_start = time.time() # Track time for data loading/filtering phase for reference and monitoring purposes
    
    train_data = load_train_data(TRAIN_JSONL) # Load the training data from the specified JSONL file (this includes all questions and their metadata, which we will filter to pure-text questions for testing the BASELINE system)
    pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]] # Filter to only include pure-text questions (this matches the capabilities of the BASELINE system, which does not handle table or image-based questions, so we focus on the subset of questions that are purely text-based for a fair evaluation of the baseline pipeline's performance on its intended question type)
    
    data_time = time.time() - data_start # Time taken for data loading/filtering phase for reference and monitoring purposes
    print(f"Loaded {len(train_data)} total questions")
    print(f"Filtered to {len(pure_text_data)} pure-text questions")
    print(f"[OK] Data loading completed in {data_time:.2f}s\n")

    # ===== PHASE 4: Single Query Test =====
    print("-" * 80)
    print("===========PHASE 4: Testing Single Query=============")
    sample_query = pure_text_data[0] # Test a single query for a quick sanity check of the pipeline before running the full evaluation (this allows us to verify that the retrieval and generation components are working end-to-end on a sample question before we run the full evaluation on multiple questions, which can save time if there are any issues that need to be addressed in the pipeline)
    
    query_start = time.time() # Track time for processing this single query through the entire pipeline for reference and monitoring purposes
    result = pipeline.run_query(sample_query["question"]) # Run the query through the pipeline (retrieval + generation)
    query_time = time.time() - query_start # Time taken for processing this single query through the entire pipeline for reference and monitoring purposes
    
    print(f"\nQuestion: {sample_query['question']}")
    print(f"Ground Truth: {sample_query['answer']}")
    print(f"Generated Answer: {result['answer']}")
    
    # Print per-query timing if available
    if 'timing' in result:
        print(f"\n  Query Timing:")
        print(f"    Retrieval:  {result['timing'].get('retrieval', 0):.4f}s")
        print(f"    Generation: {result['timing'].get('generation', 0):.4f}s")
        print(f"    Total:      {result['timing'].get('total', 0):.4f}s")
    
    print(f"\n[OK] Single query completed in {query_time:.2f}s\n")

    # ===== PHASE 5: Full Evaluation =====
    print("-" * 80)
    print("===========PHASE 5: Running Full Evaluation (Pure-Text Questions)=============")
    
    eval_start = time.time() # Track time for evaluation phase for reference and monitoring purposes
    
    metrics = pipeline.evaluate(pure_text_data[:20]) # Evaluate on the first 20 pure-text questions (this allows us to get a comprehensive evaluation of the baseline pipeline's performance on a representative subset of the pure-text questions, while keeping the evaluation time reasonable for testing purposes. In a full evaluation, we could run this on all pure-text questions or a larger subset depending on time constraints.)
    
    eval_time = time.time() - eval_start # Time taken for evaluation phase for reference and monitoring purposes
    print(f"[OK] Evaluation completed in {eval_time:.2f}s\n")
    
    # ===== SUMMARY REPORT =====
    print(f"\n{'='*80}")
    print("BASELINE SYSTEM - TIMING SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Phase Breakdown:")
    print(f"  1. Index Build/Load:          {build_time:>10.2f}s")
    print(f"  2. Component Initialization:  {init_time:>10.2f}s")
    print(f"  3. Data Loading/Filtering:    {data_time:>10.2f}s")
    print(f"  4. Single Query Test:         {query_time:>10.2f}s")
    print(f"  5. Full Evaluation (20 qs):   {eval_time:>10.2f}s")
    print(f"  " + "-" * 50)
    
    main_total = time.time() - main_start
    print(f"  TOTAL RUNTIME:                {main_total:>10.2f}s\n")
    
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
#-----------------------------------------------------------------
if __name__ == "__main__":
    main()