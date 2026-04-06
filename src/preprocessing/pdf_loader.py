
# src/data/loaders.py
# Alternative PDF processor using unstructured library

from pathlib import Path
from typing import Dict, List, Any
import logging
from unstructured.partition.auto import partition
import json
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file using unstructured library.
    Returns dict with pdf_name, full_text, and serialized blocks.
    """
    try:
        # Use fast strategy to avoid slow OCR if you only need the text/layout from native PDFs
        blocks = partition(filename=str(pdf_path), strategy="fast", languages=["eng"])
        # print("loading", pdf_path)
        
        # Serialize immediately! Returning raw unstructured elements over 
        # multiprocessing queues on Windows causes pickling issues/empty arrays
        serialized_blocks = []
        for b in blocks:
            serialized_blocks.append({
                "category": b.category,
                "text": b.text,
                "page_number": b.metadata.to_dict().get("page_number")
            })

        return {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "blocks": serialized_blocks,
        }

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None


def process_all_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory using unstructured."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    all_documents = []

    for i, pdf_path in enumerate(pdf_files):
        if i % 10 == 0:
            logger.info(f"Processing PDF {i + 1}/{len(pdf_files)}")

        doc = extract_text_from_pdf(pdf_path)

        if doc:
            all_documents.append(doc)

        # cutting it short for testing
        # if i == 3:
            # return all_documents

    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents


def process_all_pdfs_fast(pdf_dir: Path, max_workers: int = None) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory using unstructured with Multiprocessing."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1

    all_documents = []
    
    # ProcessPoolExecutor is used to bypass the GIL and utilize multiple CPU cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_text_from_pdf, p): p for p in pdf_files}
        
        for i, future in enumerate(as_completed(futures)):
            if i % 10 == 0:
                logger.info(f"Completed {i}/{len(pdf_files)} PDFs")
            
            try:
                doc = future.result()
                if doc is not None:
                    all_documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process a PDF: {e}")

    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents


def save_read_pdf_data(all_documents, path):
    # Documents are already serialized from `extract_text_from_pdf`, so we can dump them directly
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
        

def load_read_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        doc["blocks"] = [
            SimpleNamespace(
                category=b["category"],
                text=b["text"],
                metadata=SimpleNamespace(**b.get("metadata", {}))
            ) for b in doc["blocks"]
        ]
    return docs

if __name__ == "__main__":
    path = Path("./train/pdfs_train")

    # all_documents = process_all_pdfs(path)
    # save_read_pdf_data(all_documents)
    # all_documents = load_read_documents("all_documents.json")
    
    all_documents = process_all_pdfs_fast(path)
    save_read_pdf_data(all_documents, path="all_documents.json")
    