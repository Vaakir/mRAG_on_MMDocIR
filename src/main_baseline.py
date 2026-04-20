# src/main_baseline.py

import os
import sys
import logging
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append(str(Path(__file__).parent))
from config.config import BaselineConfig, AdvancedConfig
from data.data_loader import load_train_data
from pipelines.baseline_pipeline import BaselineRAGPipeline
from pipelines.advanced_pipeline import AdvancedRAGPipeline
from main import run_single_experiment

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = BaselineConfig()
    pure_text_data = load_train_data(config.TRAIN_JSONL)
    run_single_experiment(
        experiment_name="Baseline RAG",
        config=config,
        pipeline_class=BaselineRAGPipeline,
        test_data=pure_text_data[:config.EVAL_SUBSET_SIZE],
        force_rebuild=False,
        run_single_query=True
    )
    
    # how to run:
    # DAT560project> python src/main_baseline.py

    # config = AdvancedConfig()
    # train_data = load_train_data(config.TRAIN_JSONL)    
    # run_single_experiment(
    #     experiment_name=f"Advanced RAG ({config.QUERY_TECHNIQUE})",
    #     config=config,
    #     pipeline_class=AdvancedRAGPipeline,
    #     test_data=train_data[:config.EVAL_SUBSET_SIZE],
    #     force_rebuild=False,
    #     run_single_query=True
    # )
