# src/main.py

import os
import sys
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import TRAIN_JSONL, TEST_JSONL
from data.data_loader import load_train_data, load_test_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

def main():
    # Initialize pipeline
    pipeline = BaselineRAGPipeline()
    
    # Build/load index (set to False to use existing index)
    # IMPORTANT: Set to True when switching to pre-processed chunks!
    pipeline.build_index(force_rebuild=True)
    
    # Initialize components
    pipeline.initialize_components()
    
    # Test with a single query
    print("\n=== Testing Single Query ===")
    test_data = load_test_data(TEST_JSONL)
    sample_query = test_data[0]
    
    result = pipeline.run_query(sample_query["question"])
    print(f"Question: {sample_query['question']}")
    print(f"Ground Truth: {sample_query['answer']}")
    print(f"Generated Answer: {result['answer']}")
    
    # Run full evaluation
    print("\n=== Running Full Evaluation ===")
    metrics = pipeline.evaluate(test_data[:12])
    
    return metrics

if __name__ == "__main__":
    main()