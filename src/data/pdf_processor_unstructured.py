# src/data/pdf_processor_unstructured.py
# PDF processor using unstructured library with pdftotext and OCR fallbacks
# for font-encoded PDFs that produce (cid:XX) garbage and image-only PDFs

from pathlib import Path
from typing import Dict, List, Any
import logging
import subprocess
import json
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from unstructured.partition.auto import partition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_cid_garbage(text: str, threshold: float = 0.3) -> bool:
    """Return True if more than threshold fraction of chars look like (cid:XX) tokens."""
    if not text:
        return True
    cid_count = text.count("(cid:")
    return (cid_count * 8) / max(len(text), 1) > threshold


def _is_image_only(pdf_path: Path) -> bool:
    """Return True if every page has no extractable words (image-only PDF)."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:3]:  # check first 3 pages only
            if page.extract_words():
                return False
    return True


def _ocr_blocks(pdf_path: Path, dpi: int = 200) -> List[Dict[str, Any]]:
    """
    OCR an image-only PDF using pdf2image + pytesseract.
    Returns blocks in the same serialized format as unstructured.
    """
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(pdf_path), dpi=dpi)
    blocks = []
    for page_num, img in enumerate(images, 1):
        text = pytesseract.image_to_string(img, lang="eng").strip()
        if text:
            blocks.append({
                "category": "NarrativeText",
                "text": text,
                "page_number": page_num,
            })
    return blocks


def _pdftotext_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract per-page text using pdftotext (handles custom/encrypted font encodings).
    Returns blocks in the same serialized format as unstructured.
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)

    blocks = []
    for page_num in range(1, n_pages + 1):
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page_num), "-l", str(page_num),
             str(pdf_path), "-"],
            capture_output=True, text=True
        )
        text = result.stdout.strip()
        if text:
            blocks.append({
                "category": "NarrativeText",
                "text": text,
                "page_number": page_num,
            })
    return blocks


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file using unstructured library.
    Falls back to pdftotext if font-encoded garbage is detected.
    Falls back to OCR (Tesseract) if the PDF is image-only.
    Returns dict with pdf_name, pdf_path, and serialized blocks.
    """
    try:
        # Check for image-only PDF first — partition() crashes on these
        if _is_image_only(pdf_path):
            logger.warning(f"{pdf_path.name}: image-only PDF, running OCR (Tesseract)")
            serialized_blocks = _ocr_blocks(pdf_path)
        else:
            # Use fast strategy for native PDFs
            blocks = partition(filename=str(pdf_path), strategy="fast", languages=["eng"])

            # Serialize immediately — raw unstructured elements can't be pickled across processes
            serialized_blocks = []
            for b in blocks:
                serialized_blocks.append({
                    "category": b.category,
                    "text": b.text,
                    "page_number": b.metadata.to_dict().get("page_number")
                })

            # Check if unstructured produced (cid:XX) garbage (font-encoded PDFs)
            all_text = " ".join(b["text"] for b in serialized_blocks)
            if _is_cid_garbage(all_text):
                logger.warning(f"{pdf_path.name}: font-encoding garbage, falling back to pdftotext")
                serialized_blocks = _pdftotext_blocks(pdf_path)

        if not serialized_blocks:
            logger.warning(f"No content extracted from {pdf_path.name}")
            return None

        return {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "blocks": serialized_blocks,
        }

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None


def process_all_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory sequentially."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    all_documents = []
    for i, pdf_path in enumerate(pdf_files):
        if i % 10 == 0:
            logger.info(f"Processing PDF {i + 1}/{len(pdf_files)}")
        doc = extract_text_from_pdf(pdf_path)
        if doc:
            all_documents.append(doc)

    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents


def process_all_pdfs_fast(pdf_dir: Path, max_workers: int = None) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory using multiprocessing."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1

    all_documents = []
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
                pdf_path = futures[future]
                logger.error(f"Failed to process {pdf_path.name}: {type(e).__name__}: {e}")

    logger.info(f"Successfully processed {len(all_documents)} PDFs")
    return all_documents


def save_read_pdf_data(all_documents, path):
    """Save processed PDF data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)


def load_read_documents(path):
    """Load processed PDF data from JSON and convert blocks back to SimpleNamespace."""
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        doc["blocks"] = [
            SimpleNamespace(
                category=b["category"],
                text=b["text"],
                page_number=b.get("page_number"),
                metadata=SimpleNamespace()
            ) for b in doc["blocks"]
        ]
    return docs
