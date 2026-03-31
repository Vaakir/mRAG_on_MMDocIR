# src/data/pdf_processor_unstructured.py
# Alternative PDF processor using unstructured library

from pathlib import Path
from typing import Dict, List, Any
import logging
from unstructured.partition.auto import partition
from pathlib import Path
import json
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file using unstructured library.
    Returns dict with pdf_name, full_text, and page_texts.
    """
    try:
        blocks = partition(filename=str(pdf_path), languages=["eng"])
        print("loading", pdf_path)
        
        return {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "blocks": blocks,
        }

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None

# -------------------------------------------------------------------
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
# -------------------------------------------------------------------
def save_read_pdf_data(all_documents, path):
    """Save the processed PDF data to a JSON file for later loading and use."""
    serializable = [] # List to hold the serializable version of the documents (since the original blocks may contain non-serializable objects, we convert them to simple dicts for JSON serialization)
    
    # Convert the blocks to a serializable format (list of dicts) while keeping the original metadata for later use in retrieval and generation (this allows us to save the processed PDF data in a format that can be easily loaded and used for embedding, retrieval, and generation tasks without needing to reprocess the PDFs)
    for doc in all_documents:
        serializable.append({
            "pdf_name": doc["pdf_name"], # Add the PDF name
            "pdf_path": doc["pdf_path"], # Add the PDF path
            "blocks": [{ # Convert each block to a dict with category, text, and page number (if available) for serialization while keeping the original metadata for later use in retrieval and generation
                "category": b.category,
                "text": b.text,
                "page_number": b.metadata.to_dict().get("page_number")
            } for b in doc["blocks"]]
        })

    # Save the serializable data to a JSON file (this allows us to persist the processed PDF data in a format that can be easily loaded and used for embedding, retrieval, and generation tasks without needing to reprocess the PDFs)
    with open("all_documents.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
        
# -------------------------------------------------------------------
def load_read_documents(path):
    """Load the processed PDF data from a JSON file and convert it back to the 
    original format for use in embedding, retrieval, and generation tasks."""
    
    # Load the JSON file, which contains pre-processed chunks (list of documents with their chunks)
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # Convert the loaded data back to the original format (with blocks as SimpleNamespace) 
    for doc in docs:
        doc["blocks"] = [
            SimpleNamespace(
                category=b["category"],
                text=b["text"],
                metadata=SimpleNamespace(**b.get("metadata", {}))
            ) for b in doc["blocks"]
        ]
    return docs
# -------------------------------------------------------------------
if __name__ == "__main__":
    #path = Path("../train/pdfs_train")

    # all_documents = process_all_pdfs(path)
    # save_read_pdf_data(all_documents)
    all_documents = load_read_documents("all_documents.json")
    