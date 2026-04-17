# src/main_baseline.py

import os
import sys
import time
import logging
import pandas as pd
from contextlib import contextmanager
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import BaselineConfig
from utils.timer import MetricsTracker
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline

logger = logging.getLogger(__name__)

def main(force_rebuild: bool = False):
    config = BaselineConfig()
    pipeline = BaselineRAGPipeline(config)
    tracker = MetricsTracker(logger)
    log_and_time = tracker.log_and_time

    with log_and_time('Total Pipeline Runtime'):
        
        with log_and_time('Index Build/Load'):
            pipeline.build_index(force_rebuild=force_rebuild)

        with log_and_time('Component Initialization'):
            pipeline.initialize_components()

        with log_and_time('Data Loading/Filtering'):
            pure_text_data = load_train_data(config.TRAIN_JSONL, pure_text=True)

        with log_and_time('Single Query Test'):
            result = pipeline.run_query(pure_text_data[0]["question"])
            logger.info(f"Test -> Q: {pure_text_data[0]['question'][:50]}... | Gen: {result['answer'][:50]}...")

        with log_and_time('Full Evaluation'):
            eval_metrics = pipeline.evaluate(pure_text_data[:config.EVAL_SUBSET_SIZE])

    flat_metrics = tracker.flatten_eval_metrics("Baseline RAG", eval_metrics)
    tracker.save_to_csv(flat_metrics, config.RESULTS_CSV)
    tracker.save_to_csv(tracker.timing_data, config.BASELINE_TIME_CSV)
    tracker.print_metrics("FINAL EVALUATION METRICS", flat_metrics)

    return flat_metrics

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # (force_rebuild) Only needed if the local_qdrant database does not exist (need only run once)    
    main(force_rebuild=False)
    
    # how to run:
    # DAT560project> python -m .\src\main_baseline.py
