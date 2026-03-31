# src/main.py

import os
import sys
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import TRAIN_JSONL
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

def main():
    # Initialize pipeline
    pipeline = BaselineRAGPipeline()
    
    # Build/load index (set to False to use existing index)
    # IMPORTANT: Set to True when switching to pre-processed chunks!
    pipeline.build_index(force_rebuild=False) # 'True' if you want to rebuild the index from raw PDFs, 'False' to load existing index (must be built at least once)

    # Initialize components
    pipeline.initialize_components()

    # Filter for pure-text questions only (since the baseline is text-only RAG)
    train_data = load_train_data(TRAIN_JSONL)
    pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]]
    print(f"\nFiltered to {len(pure_text_data)} pure-text questions (out of {len(train_data)} total)")

    # Test with a single query
    print("\n=== Testing Single Query ===")
    sample_query = pure_text_data[0]

    result = pipeline.run_query(sample_query["question"]) # Run the query through the pipeline (retrieval + generation)
    print(f"Question: {sample_query['question']}")
    print(f"Ground Truth: {sample_query['answer']}")
    print(f"Generated Answer: {result['answer']}")

    # Run full evaluation on pure-text subset
    print("\n=== Running Full Evaluation (Pure-Text Questions) ===")
    metrics = pipeline.evaluate(pure_text_data[:20])
    
    return metrics

if __name__ == "__main__":
    main()