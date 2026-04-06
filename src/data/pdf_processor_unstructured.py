# src/data/pdf_processor_unstructured.py
# Alternative PDF processor using unstructured library

from pathlib import Path
from typing import Dict, List, Any
import logging
from unstructured.partition.auto import partition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file using unstructured library.
    Returns dict with pdf_name, full_text, and page_texts.
    """
    result = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "full_text": "",
        "pages": []  # List of {page_num, text}
    }
    
    try:
        # Use unstructured to partition the PDF into semantic blocks
        blocks = partition(filename=str(pdf_path))
        
        # Group blocks by page (if page info is available)
        current_page = 1
        page_text = []
        
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # Try to get page number from metadata
            page_num = getattr(block.metadata, 'page_number', None) if hasattr(block, 'metadata') else None
            
            if page_num and page_num != current_page:
                # Save previous page
                if page_text:
                    result["pages"].append({
                        "page_num": current_page,
                        "text": " ".join(page_text)
                    })
                    page_text = []
                current_page = page_num
            
            page_text.append(text)
            result["full_text"] += text + " "
        
        # Add last page
        if page_text:
            result["pages"].append({
                "page_num": current_page,
                "text": " ".join(page_text)
            })
        
        # If no page info available, treat as single page
        if not result["pages"] and result["full_text"]:
            result["pages"].append({
                "page_num": 1,
                "text": result["full_text"]
            })
            
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    
    return result

def process_all_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory using unstructured."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_documents = []
    for i, pdf_path in enumerate(pdf_files):
        if i % 10 == 0:
            logger.info(f"Processing PDF {i+1}/{len(pdf_files)}")
        doc = extract_text_from_pdf(pdf_path)
        if doc["full_text"].strip():  # Only add if text was extracted
            all_documents.append(doc)
    
    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents
