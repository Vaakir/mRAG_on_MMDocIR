# The preprocessing pipeline runs in two stages:
# 1. PDF extraction + chunking  (pdf_loader.py, pdf_chunker.py)
# 2. Multimodal index building  (build_multimodal_indexes.py)
#    — one Qdrant collection per chunking strategy, including page images,
#      figures, and evidence crops.
import logging
import argparse
from config.config import BaselineConfig
from utils.timer import MetricsTracker
from preprocessing.pdf_chunker import chunk_and_save_pdf_data
from preprocessing.pdf_loader import (
    process_all_pdfs,
    save_read_pdf_data,
    load_read_documents,
)
from preprocessing.build_multimodal_indexes import build_for_strategy, STRATEGIES

logger = logging.getLogger(__name__)


def preprocessing_pipeline(skip_chunking: bool = False, skip_indexing: bool = False, force_rebuild: bool = False):
    """
    Full preprocessing pipeline:
      Stage 1 — PDF extraction + all chunking strategies  (skippable with skip_chunking)
      Stage 2 — Multimodal Qdrant index per strategy      (skippable with skip_indexing)
    """
    config = BaselineConfig()
    tracker = MetricsTracker(logger)
    log_and_time = tracker.log_and_time

    chunk_result = None
    all_documents = None

    with log_and_time('Total Pipeline Runtime'):

        # ── Stage 1: PDF extraction + chunking ────────────────────────────
        if not skip_chunking:
            with log_and_time('Extracting PDFs using docling'):
                all_documents = process_all_pdfs(config.PDFS_DIR)

            if not all_documents:
                raise ValueError("PDF extraction returned no documents.")

            with log_and_time(f"Saving extracted JSON to {config.PREPROCESSED_DOCUMENTS_FILE}"):
                save_read_pdf_data(all_documents, path=config.PREPROCESSED_DOCUMENTS_FILE)

            with log_and_time('Loading documents for chunking'):
                all_documents = load_read_documents(config.PREPROCESSED_DOCUMENTS_FILE)

            with log_and_time('Running chunking methods and saving outputs'):
                chunk_result = chunk_and_save_pdf_data(
                    all_documents, output_dir=config.PREPROCESSED_DATA_DIR
                )
        else:
            logger.info("Skipping Stage 1 (PDF extraction + chunking)")

        # ── Stage 2: Multimodal index building ────────────────────────────
        if not skip_indexing:
            for strategy in STRATEGIES:
                with log_and_time(f"Building multimodal index: {strategy}"):
                    build_for_strategy(strategy, force_rebuild=force_rebuild)
        else:
            logger.info("Skipping Stage 2 (multimodal index building)")

    tracker.print_timing_summary()
    tracker.save_to_csv(tracker.timing_data, config.PREPROCESSING_TIME_CSV)

    return {
        "status": "success",
        "processed_count": len(all_documents) if all_documents else "skipped",
        "chunk_info": chunk_result,
        "timing": tracker.timing_data,
    }


if __name__ == "__main__":

    result = preprocessing_pipeline(
        skip_chunking=True,
        skip_indexing=False,
        force_rebuild=False,
    )
    print(f"Preprocessing pipeline completed: {result['status']}")

    # how to run:
    # DAT560project/src> python -m pipelines.preprocessing_pipeline