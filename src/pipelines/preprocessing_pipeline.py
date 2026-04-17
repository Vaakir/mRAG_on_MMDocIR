# The preprocessing pipeline calls upon the following preprocessing files
# 1. pdf_loader.py
# 2. pdf_chunker.py
import logging
import time
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from config.config import BaselineConfig
from utils.timer import MetricsTracker
from preprocessing.pdf_chunker import chunk_and_save_pdf_data
from preprocessing.pdf_loader import (
    process_all_pdfs,
    save_read_pdf_data,
    load_read_documents,
)

logger = logging.getLogger(__name__)


def preprocessing_pipeline():
    """Runs the preprocessing pipeline."""
    config = BaselineConfig()
    tracker = MetricsTracker(logger)
    log_and_time = tracker.log_and_time

    with log_and_time('Total Pipeline Runtime'):

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

    tracker.print_timing_summary()

    tracker.save_to_csv(tracker.timing_data, config.PREPROCESSING_TIME_CSV)

    return {
        "status": "success",
        "processed_count": len(all_documents),
        "chunk_info": chunk_result,
        "timing": tracker.timing_data
    }


if __name__ == "__main__":
    result = preprocessing_pipeline()
    print(f"Preprocessing pipeline completed successfully: {result}")

    # how to run:
    # DAT560project/src> python -m pipelines.preprocessing_pipeline
