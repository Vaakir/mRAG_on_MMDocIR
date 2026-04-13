import os
import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Fix OpenMP library conflict and prevent TF/Numpy 2.0 version crash in transformers
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['USE_TF'] = '0'

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent))

from config.config import BaselineConfig, AdvancedConfig, RESULTS_CSV
from pipelines.baseline_pipeline import BaselineRAGPipeline
from pipelines.advanced_pipeline import AdvancedRAGPipeline
from data.data_loader import load_train_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_results_to_csv(experiment_name: str, config_type: str, timings: Dict[str, float], metrics: Dict[str, Any]):
    """Flatten metrics and timings and save to CSV."""
    # Flatten the result dictionary for CSV
    row = {
        "Experiment Name": experiment_name,
        "System Type": config_type,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Total Runtime (s)": round(timings.get('Total Runtime', 0), 2),
        "Eval Runtime (s)": round(timings.get('Evaluation Time', 0), 2)
    }

    # Add Retrieval Metrics
    if metrics and 'retrieval' in metrics:
        for k, v in metrics['retrieval'].items():
            row[f"Retrieval {k}"] = round(v, 4) if isinstance(v, float) else v

    # Add Generation Metrics
    if metrics and 'generation' in metrics:
        for k, v in metrics['generation'].items():
            row[f"Generation {k}"] = round(v, 4) if isinstance(v, float) else v

    df_new = pd.DataFrame([row])

    if RESULTS_CSV.exists():
        df_existing = pd.read_csv(RESULTS_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Results for '{experiment_name}' saved to {RESULTS_CSV}")


def run_single_experiment(experiment_name: str, config, pipeline_class, test_data: List[Dict[str, Any]]):
    """Run a single pipeline configuration and log metrics."""
    logger.info(f"\n{'='*80}\nSTARTING EXPERIMENT: {experiment_name}\n{'='*80}")
    
    timings = {}
    main_start = time.time()
    
    pipeline = pipeline_class(config)
    
    # Init components
    logger.info(f"[{experiment_name}] Building index / initializing...")
    build_start = time.time()
    pipeline.build_index(force_rebuild=False)
    timings['Index Build Runtime (s)'] = time.time() - build_start
    
    pipeline.initialize_components()
    
    # Run evaluation
    logger.info(f"[{experiment_name}] Evaluating...")
    eval_start = time.time()
    
    # Subsample data if needed (the config parameter handles this, but passing full test set to evaluate())
    # Make sure we use `use_technique=True` for Advanced Pipeline if supported.
    if isinstance(pipeline, AdvancedRAGPipeline):
        metrics = pipeline.evaluate(test_data, use_technique=True)
    else:
        metrics = pipeline.evaluate(test_data)
        
    timings['Evaluation Time'] = time.time() - eval_start
    timings['Total Runtime'] = time.time() - main_start

    # Extract pipeline type string to save in CSV
    pipeline_type = "Baseline" if isinstance(pipeline, BaselineRAGPipeline) else "Advanced"

    # Save
    save_results_to_csv(experiment_name, pipeline_type, timings, metrics)
    return metrics


def main():
    # 1. Load Data
    # For fair comparison, we use the same validation subset from train.jsonl (or test.jsonl)
    logger.info("Loading evaluation data...")
    base_config = BaselineConfig()
    train_data = load_train_data(base_config.TRAIN_JSONL)
    
    # Filter for pure text for baseline comparisons
    pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]]
    logger.info(f"Loaded {len(pure_text_data)} text questions for evaluation.")
    
    # IMPORTANT: Adjust subset size for experiments. 
    # Use a small number like 5-10 for fast testing, or 50+ for final rigorous evaluation.
    EVAL_SUBSET_SIZE = 2

    # Helper function to extract a primary metric score to pick the "best"
    def get_score(metrics):
        if not metrics: return -1
        # Let's use generation token_f1 or exact_match as the primary decider, fallback to retrieval MRR
        if 'generation' in metrics and 'token_f1' in metrics['generation']:
            return metrics['generation']['token_f1']
        elif 'retrieval' in metrics and 'mrr' in metrics['retrieval']:
            return metrics['retrieval']['mrr']
        return -1

    # ==========================================
    # EXPERIMENT 1: BASELINE
    # ==========================================
    baseline_cfg = BaselineConfig(EVAL_SUBSET_SIZE=EVAL_SUBSET_SIZE)
    run_single_experiment("1_Baseline", baseline_cfg, BaselineRAGPipeline, pure_text_data)

    # ==========================================
    # EXPERIMENT 2: CHUNKING ABLATIONS (Advanced)
    # ==========================================
    chunking_strategies = ["fixed_size", "sliding_window", "semantic", "hierarchical"]
    best_chunking = "fixed_size"
    best_chunking_score = -1

    for strategy in chunking_strategies:
        cfg = AdvancedConfig(
            EVAL_SUBSET_SIZE=EVAL_SUBSET_SIZE,
            CHUNKING_STRATEGY=strategy,
            PREPROCESSED_CHUNKS_FILE=str(Path(baseline_cfg.SRC_DIR) / "data" / "preprocessed" / f"chunks_{strategy}.json"),
            VECTOR_DB_COLLECTION=f"bge_collection_{strategy}"
        )
        try:
            metrics = run_single_experiment(f"2_Chunking_{strategy}", cfg, AdvancedRAGPipeline, pure_text_data)
            score = get_score(metrics)
            if score > best_chunking_score:
                best_chunking_score = score
                best_chunking = strategy
        except Exception as e:
            logger.error(f"Experiment 2_Chunking_{strategy} failed: {e}")

    logger.info(f"\n*** Winner from Chunking: {best_chunking.upper()} (Score: {best_chunking_score:.4f}) ***\n")

    # ==========================================
    # EXPERIMENT 3: QUERY PROCESSING ABLATIONS
    # ==========================================
    query_techniques = [
        "standard", "multi_query", "rag_fusion", "step_back", 
        "hyde", "query_decomposition", "query_rewriting", "query_expansion"
    ]
    best_technique = "standard"
    best_technique_score = -1

    for tech in query_techniques:
        cfg = AdvancedConfig(
            EVAL_SUBSET_SIZE=EVAL_SUBSET_SIZE,
            CHUNKING_STRATEGY=best_chunking,
            PREPROCESSED_CHUNKS_FILE=str(Path(baseline_cfg.SRC_DIR) / "data" / "preprocessed" / f"chunks_{best_chunking}.json"),
            VECTOR_DB_COLLECTION=f"bge_collection_{best_chunking}",
            QUERY_TECHNIQUE=tech
        )
        try:
            metrics = run_single_experiment(f"3_QueryTech_{tech}", cfg, AdvancedRAGPipeline, pure_text_data)
            score = get_score(metrics)
            if score > best_technique_score:
                best_technique_score = score
                best_technique = tech
        except Exception as e:
            logger.error(f"Experiment 3_QueryTech_{tech} failed: {e}")

    logger.info(f"\n*** Winner from Query Techniques: {best_technique.upper()} (Score: {best_technique_score:.4f}) ***\n")

    # ==========================================
    # EXPERIMENT 4: PROMPTING STRATEGY ABLATIONS
    # ==========================================
    prompting_strategies = ["standard", "few_shot", "role", "cot", "ensemble"]
    
    for prompt_strat in prompting_strategies:
        cfg = AdvancedConfig(
            EVAL_SUBSET_SIZE=EVAL_SUBSET_SIZE,
            CHUNKING_STRATEGY=best_chunking,
            PREPROCESSED_CHUNKS_FILE=str(Path(baseline_cfg.SRC_DIR) / "data" / "preprocessed" / f"chunks_{best_chunking}.json"),
            VECTOR_DB_COLLECTION=f"bge_collection_{best_chunking}",
            QUERY_TECHNIQUE=best_technique,
            PROMPTING_STRATEGY=prompt_strat
        )
        try:
            run_single_experiment(f"4_Prompting_{prompt_strat}", cfg, AdvancedRAGPipeline, pure_text_data)
        except Exception as e:
            logger.error(f"Experiment 4_Prompting_{prompt_strat} failed: {e}")

    # ==========================================
    # EXPERIMENT 5: MULTIMODAL VS TEXT
    # ==========================================
    # Example: you'd use a specific multimodal config here, using CLIP embeddings
    mm_cfg = AdvancedConfig(
        EVAL_SUBSET_SIZE=EVAL_SUBSET_SIZE,
        EMBEDDING_MODEL="jinaai/jina-clip-v2",
        VECTOR_DB_COLLECTION="baseline_documents_jina"
    )
    try:
        run_single_experiment("5_Multimodal_Retrieval", mm_cfg, AdvancedRAGPipeline, pure_text_data)
    except Exception as e:
        logger.error(f"Experiment 5_Multimodal_Retrieval failed: {e}")

    logger.info("\nAll experiments completed! Check experiments_results.csv for details.")

if __name__ == "__main__":
    main()
