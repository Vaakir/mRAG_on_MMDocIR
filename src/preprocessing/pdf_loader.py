
# src/preprocessing/pdf_loader.py
# PDF processor using docling — auto-detects and removes page headers/footers by label
# Falls back to forced-OCR docling when text extraction produces undecodable font garbage

from pathlib import Path
from typing import Dict, List, Any
import logging
import json
import re
from types import SimpleNamespace

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat, DocItemLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# /G-encoded glyphs: /G44 → chr(0x44) == 'D'  (plain ASCII hex)
_GLYPH_RE = re.compile(r"/G([0-9A-Fa-f]{2})")
# CID refs from other parsers — no font table, cannot decode
_CID_RE = re.compile(r"\(cid:\d+\)")

SKIP_LABELS = {DocItemLabel.PAGE_FOOTER, DocItemLabel.PAGE_HEADER}


def _build_converter() -> DocumentConverter:
    """Build a DocumentConverter with GPU if available, CPU otherwise."""
    try:
        import torch
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    except ImportError:
        device = AcceleratorDevice.CPU

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=device,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# Module-level singleton — models load once and are reused across all PDFs
_converter: DocumentConverter = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        logger.info("Initialising docling DocumentConverter...")
        _converter = _build_converter()
    return _converter


def _docling_label_to_category(label: DocItemLabel) -> str:
    if label == DocItemLabel.TITLE:
        return "Title"
    if label == DocItemLabel.SECTION_HEADER:
        return "Header"
    return "NarrativeText"


def _decode_glyph_text(text: str) -> str:
    """Decode /Gxx hex sequences to their ASCII characters."""
    return _GLYPH_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _has_cid_corruption(blocks: list) -> bool:
    """Return True if >20% of blocks still contain undecoded (cid:N) sequences."""
    if not blocks:
        return False
    corrupted = sum(1 for b in blocks if _CID_RE.search(b.get("text", "")))
    return corrupted / len(blocks) > 0.2


def _docling_result_to_blocks(result) -> list:
    serialized_blocks = []
    for item, _ in result.document.iterate_items():
        if item.label in SKIP_LABELS:
            continue

        # Tables: item.text is None — export as markdown instead
        if item.label == DocItemLabel.TABLE:
            try:
                text = item.export_to_markdown().strip()
            except Exception:
                text = item.text.strip() if item.text else ""
        else:
            text = item.text.strip() if hasattr(item, "text") and item.text else ""

        if not text:
            continue
        text = _decode_glyph_text(text)
        page_number = item.prov[0].page_no if item.prov else None
        serialized_blocks.append({
            "category": _docling_label_to_category(item.label),
            "text": text,
            "page_number": page_number,
        })
    return serialized_blocks


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text from a PDF using docling.
    Page headers and footers are excluded automatically by label.
    /Gxx glyph sequences are decoded in-place as ASCII hex.
    Returns dict with pdf_name, pdf_path, and serialized blocks.
    """
    pdf_path = Path(pdf_path)
    try:
        result = _get_converter().convert(str(pdf_path))
        blocks = _docling_result_to_blocks(result)

        if _has_cid_corruption(blocks):
            logger.warning(
                f"{pdf_path.name} still has (cid:) sequences after decoding — "
                "font table unavailable, text may be partial"
            )

        return {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "blocks": blocks,
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None


def process_all_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory using docling."""
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


# Alias expected by the preprocessing pipeline
process_all_pdfs_fast = process_all_pdfs


def save_read_pdf_data(all_documents, path):
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
                page_number=b.get("page_number")
            ) for b in doc["blocks"]
        ]
    return docs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.config import BaselineConfig

    cfg = BaselineConfig()
    logger.info(f"Processing PDFs from: {cfg.PDFS_DIR}")
    all_documents = process_all_pdfs(cfg.PDFS_DIR)
    save_read_pdf_data(all_documents, path=cfg.PREPROCESSED_DOCUMENTS_FILE)
    logger.info(f"Saved {len(all_documents)} documents to {cfg.PREPROCESSED_DOCUMENTS_FILE}")
