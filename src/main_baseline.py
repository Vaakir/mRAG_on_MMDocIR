# src/main_baseline.py

import os
import sys
import time
import logging
from contextlib import contextmanager
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import BaselineConfig
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    metrics = {'timing': {}}

    @contextmanager
    def time_phase(name):
        start = time.time()
        yield
        metrics['timing'][name] = time.time() - start

    with time_phase('Total Pipeline Runtime'):
        logger.info("Starting BASELINE RAG SYSTEM")
        config = BaselineConfig()
        pipeline = BaselineRAGPipeline(config)
        
        with time_phase('Index Build/Load'):
            pipeline.build_index()

        with time_phase('Component Initialization'):
            pipeline.initialize_components()

        with time_phase('Data Loading/Filtering'):
            train_data = load_train_data(config.TRAIN_JSONL)
            pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]]
            logger.info(f"Loaded {len(train_data)} qs, filtered to {len(pure_text_data)} pure-text")

        with time_phase('Single Query Test'):
            sample_query = pure_text_data[0]
            result = pipeline.run_query(sample_query["question"])
            logger.info(f"Test -> Q: {sample_query['question'][:50]}... | Gen: {result['answer'][:50]}...")

        with time_phase('Full Evaluation (20 qs)'):
            eval_metrics = pipeline.evaluate(pure_text_data[:20])

        # Merge new metrics without clobbering inner timing dict
        metrics.update({k: v for k, v in eval_metrics.items() if k != 'timing'})
        if 'timing' in eval_metrics:
            metrics['timing'].update(eval_metrics['timing'])

    # Clean Printing
    print(f"\n{'='*50}\n FINAL EVALUATION METRICS\n{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n--- {key.upper()} ---")
            for sub_key, sub_val in value.items():
                print(f"  {sub_key:<30}: {sub_val:.4f}" if isinstance(sub_val, float) else f"  {sub_key:<30}: {sub_val}")
        else:
            print(f"{key:<32}: {value:.4f}" if isinstance(value, float) else f"{key:<32}: {value}")
    print(f"\n{'='*50}\n")
    
    return metrics

if __name__ == "__main__":
    main()