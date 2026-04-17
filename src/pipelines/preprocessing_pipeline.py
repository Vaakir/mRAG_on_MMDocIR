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
    timing_data = {"Phase": [], "Duration (seconds)": [], "Timestamp": []}

    @contextmanager
    def time_phase(name):
        logger.info(f"Starting: {name}...")
        start = time.time()
        yield
        duration = time.time() - start
        
        timing_data["Phase"].append(name)
        timing_data["Duration (seconds)"].append(round(duration, 4))
        timing_data["Timestamp"].append(time.strftime('%Y-%m-%d %H:%M:%S'))

    with time_phase('Total Pipeline Runtime'):
        # 1. pdf_loader.py
        with time_phase('Extracting PDFs using docling'):
            all_documents = process_all_pdfs(config.PDFS_DIR)

        if not all_documents:
            raise ValueError("PDF extraction returned no documents.")

        with time_phase(f"Saving extracted JSON to {config.PREPROCESSED_DOCUMENTS_FILE}"):
            save_read_pdf_data(all_documents, path=config.PREPROCESSED_DOCUMENTS_FILE)

        # 2. pdf_chunker.py
        with time_phase('Loading documents for chunking'):
            all_documents = load_read_documents(config.PREPROCESSED_DOCUMENTS_FILE)

        with time_phase('Running chunking methods and saving outputs'):
            chunk_result = chunk_and_save_pdf_data(
                all_documents, output_dir=config.PREPROCESSED_DATA_DIR
            )

    # 3. save time taken
    csv_file = config.PREPROCESSING_TIME_CSV
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(timing_data)
    df.to_csv(csv_file, mode='a', index=False, header=not csv_file.exists())
    
    for _, row in df.iterrows():
        logger.info(f"{row['Phase']:<30}: {row['Duration (seconds)']} seconds")

    return {
        "status": "success",
        "processed_count": len(all_documents),
        "chunk_info": chunk_result,
        "timing": timing_data
    }


if __name__ == "__main__":
    result = preprocessing_pipeline()
    print(f"Preprocessing pipeline completed successfully: {result}")

    # how to run:
    # DAT560project/src> python -m pipelines.preprocessing_pipeline
