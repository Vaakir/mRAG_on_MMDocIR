# Using unstructured library for PDF processing

import os
from pathlib import Path
from typing import Dict, List, Any
import logging

# Skip HEIF support (requires MSYS2 compiler on Windows)
# Most PDFs don't have HEIF images anyway
os.environ['UNSTRUCTURED_SKIP_HEIF'] = 'true'

# Try pdfplumber first (simpler, more reliable on Windows)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Try unstructured (better structure detection but needs dependencies)
UNSTRUCTURED_AVAILABLE = False
try:
    # Try to import pi_heif first to check if it's available
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    if "pi_heif" in str(e):
        logging.info("Unstructured requires pi_heif - using pdfplumber instead")
    UNSTRUCTURED_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextBlock:
    """Simple text block class to mimic unstructured Element."""
    def __init__(self, text, category="NarrativeText", metadata=None):
        self.text = text
        self.category = category
        self.metadata = metadata or {}


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file using unstructured library.
    Returns dict with pdf_name, pdf_path, and blocks (structured elements).
    
    Uses unstructured's semantic block detection for better structure preservation.
    Falls back to pdfplumber if unstructured has dependency issues.
    """
    # Since pi_heif is not available, use pdfplumber directly
    if not UNSTRUCTURED_AVAILABLE and PDFPLUMBER_AVAILABLE:
        return extract_with_pdfplumber(pdf_path)
    
    try:
        # Try unstructured first (if available)
        if UNSTRUCTURED_AVAILABLE:
            try:
                # Partition PDF into semantic blocks
                blocks = partition(filename=str(pdf_path), languages=["eng"])
                
                if not blocks:
                    logger.warning(f"No content from {pdf_path.name}")
                    return None
                
                return {
                    "pdf_name": pdf_path.name,
                    "pdf_path": str(pdf_path),
                    "blocks": blocks
                }
            except Exception as e:
                # If unstructured fails, try pdfplumber
                if "pi_heif" in str(e) or "heif" in str(e).lower():
                    if PDFPLUMBER_AVAILABLE:
                        return extract_with_pdfplumber(pdf_path)
                raise
        
        # If unstructured not available, use pdfplumber
        elif PDFPLUMBER_AVAILABLE:
            return extract_with_pdfplumber(pdf_path)
        
        else:
            logger.error(
                "No PDF library available. "
                "Install 'unstructured[pdf]' or 'pdfplumber'"
            )
            return None

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None


def extract_with_pdfplumber(pdf_path: Path) -> Dict[str, Any]:
    """
    Fallback extraction using pdfplumber.
    Converts text to block format compatible with unstructured.
    """
    with pdfplumber.open(pdf_path) as pdf:
        all_blocks = []
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                # Create a simple text block
                block = TextBlock(
                    text=text,
                    category="NarrativeText",
                    metadata={"page_number": page_num}
                )
                all_blocks.append(block)
        
        if not all_blocks:
            return None
        
        return {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "blocks": all_blocks
        }

def process_all_pdfs(pdf_dir: Path, max_pdfs: int = None) -> List[Dict[str, Any]]:
    """
    Process all PDFs in a directory using unstructured.
    
    Args:
        pdf_dir: Directory containing PDF files
        max_pdfs: Maximum number of PDFs to process (None = all)
    
    Returns list of documents, each with:
    - pdf_name: filename
    - pdf_path: full path
    - blocks: list of structured elements from unstructured
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    # Limit number of PDFs if specified
    if max_pdfs is not None:
        pdf_files = pdf_files[:max_pdfs]
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    all_documents = []
    for i, pdf_path in enumerate(pdf_files):
        if i % 10 == 0:
            logger.info(f"Processing PDF {i+1}/{len(pdf_files)}")
        doc = extract_text_from_pdf(pdf_path)
        if doc:
            all_documents.append(doc)
    
    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents