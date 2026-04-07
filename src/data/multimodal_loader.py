# src/data/multimodal_loader.py
# Loads page images from the project_collection directory and creates image chunks.

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_page_images(
    page_images_dir: Path,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Scan page_images_{split}/ and return a flat list of image records.

    Each record:
        {
            "pdf_name":   "DSA-278777.pdf",
            "page_num":   2,              # 0-indexed as stored in filename
            "image_path": "/abs/path/DSA-278777_2.jpg",
            "chunk_type": "image",
            "chunk_id":   "DSA-278777_2",
        }

    Naming convention in dataset: {pdf_name_no_ext}_{page_num}.jpg
    """
    page_images_dir = Path(page_images_dir)
    if not page_images_dir.exists():
        raise FileNotFoundError(f"Page images directory not found: {page_images_dir}")

    records = []
    for img_path in sorted(page_images_dir.glob("*.jpg"), key=lambda p: (p.stem.rsplit("_", 1)[0], int(p.stem.rsplit("_", 1)[1]) if p.stem.rsplit("_", 1)[1].isdigit() else 0)):
        stem = img_path.stem  # e.g. "DSA-278777_2"

        # Split on last underscore to separate pdf_name from page_num
        last_underscore = stem.rfind("_")
        if last_underscore == -1:
            logger.warning(f"Skipping unexpected filename: {img_path.name}")
            continue

        pdf_stem = stem[:last_underscore]           # "DSA-278777"
        page_str = stem[last_underscore + 1:]       # "2"

        try:
            page_num = int(page_str)
        except ValueError:
            logger.warning(f"Cannot parse page number from: {img_path.name}")
            continue

        records.append({
            "pdf_name":   pdf_stem + ".pdf",
            "page_num":   page_num,
            "image_path": str(img_path.resolve()),
            "chunk_type": "image",
            "chunk_id":   stem,
        })

    logger.info(f"Loaded {len(records)} page images from {page_images_dir}")
    return records


def group_images_by_pdf(image_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group image records by pdf_name. Useful for sliding window chunking."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in image_records:
        grouped.setdefault(rec["pdf_name"], []).append(rec)
    # Sort pages within each PDF
    for pdf_name in grouped:
        grouped[pdf_name].sort(key=lambda r: r["page_num"])
    return grouped
