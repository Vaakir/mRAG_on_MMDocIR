# File that calls upon the following preprocessing files
# 1. pdf_loader.py
# 2. pdf_chunker.py
import logging
from pathlib import Path
from typing import Any, Literal

from config.config import DATA_DIR, PDF_DIR, ALL_DOCUMENTS_FILE
from preprocessing.pdf_loader import (
    process_all_pdfs,
    process_all_pdfs_fast,
    save_read_pdf_data,
    load_read_documents,
)
from preprocessing.pdf_chunker import Chunking, chunk_and_save_pdf_data

logger = logging.getLogger(__name__)


def preprocessing_pipeline(
    reading_method: Literal["standard", "multiprocessing"] = "multiprocessing",
):
    """
    Runs the preprocessing pipeline.
    """
    # 1. pdf_loader.py
    reading_method_map = {
        "standard": process_all_pdfs,
        "multiprocessing": process_all_pdfs_fast,
    }
    
    if reading_method not in reading_method_map.keys():
        raise ValueError(f"Reading method is not among {list(reading_method_map.keys())}")
    
    logger.info(f"Extracting PDFs using {reading_method}...")
    all_documents = reading_method_map[reading_method](PDF_DIR)

    if not all_documents:
        raise ValueError("PDF extraction returned no documents.")

    logger.info(f"Saving extracted JSON to {ALL_DOCUMENTS_FILE}...")
    save_read_pdf_data(all_documents, path=ALL_DOCUMENTS_FILE)

    # 2. pdf_chunker.py
    logger.info("Loading documents for chunking...")
    all_documents = load_read_documents(ALL_DOCUMENTS_FILE)

    logger.info("Running chunking methods and saving outputs...")
    chunk_result = chunk_and_save_pdf_data(all_documents, output_dir=DATA_DIR)

    return {
        "status": "success",
        "processed_count": len(all_documents),
        "chunk_info": chunk_result,
    }


if __name__ == "__main__":
    preprocessing_pipeline(reading_method="ok")
    # result = preprocessing_pipeline(reading_method="multiprocessing")
    # print(f"Pipeline completed successfully: {result}")
