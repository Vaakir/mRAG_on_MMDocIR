# src/preprocessing/image_processor.py
# Builds chunk lists for page images and evidence crops.
# Page images  → embedded with CLIP image encoder (visual content)
# Evidence crops → embedded with question text (question is the retrieval key,
#                  image path is what gets passed to the VLM at generation time)
#
# Paths stored in Qdrant are RELATIVE to DATA_DIR so the same index works
# on any machine (local Mac, Colab, etc.).  Callers resolve them at runtime.

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# JSONL evidence_images paths contain a "crops/" segment that does not exist on
# disk — files sit directly in the question folder.
_CROPS_RE = re.compile(r"/crops/")


def _resolve_evidence_path(base_dir: Path, relative_path: str) -> Path:
    """
    Resolve an evidence image path from the JSONL.
    Strips the leading "images_train/" or "images_test/" prefix and the
    spurious "crops/" segment that appears in the JSONL but not on disk.
    """
    parts = Path(relative_path).parts
    if parts and parts[0].startswith("images_"):
        relative_path = str(Path(*parts[1:]))
    relative_path = _CROPS_RE.sub("/", relative_path)
    return base_dir / relative_path


def load_page_image_chunks(page_images_dir: Path, data_dir: Path) -> List[Dict[str, Any]]:
    """
    Build a chunk list from a directory of page screenshot JPGs.
    Naming convention: {doc_name}_{page_num}.jpg

    image_path is stored as a path RELATIVE to data_dir so the index is
    portable across machines.

    Returns a list of dicts:
        type        : "page_image"
        image_path  : path relative to data_dir (string)
        doc_name    : document name (without page suffix)
        page_num    : int page number
        text        : short label used as fallback display text
    """
    page_images_dir = Path(page_images_dir)
    data_dir = Path(data_dir)
    chunks = []

    jpg_files = sorted(page_images_dir.glob("*.jpg"))
    logger.info(f"Found {len(jpg_files)} page images in {page_images_dir}")

    for jpg in jpg_files:
        stem = jpg.stem  # e.g. "05-03-18-political-release_3"
        last_underscore = stem.rfind("_")
        if last_underscore == -1:
            continue
        doc_name = stem[:last_underscore]
        try:
            page_num = int(stem[last_underscore + 1:])
        except ValueError:
            continue

        rel_path = jpg.relative_to(data_dir)  # e.g. train/page_images_train/foo_3.jpg

        chunks.append({
            "type": "page_image",
            "image_path": str(rel_path),   # relative — portable
            "doc_name": doc_name,
            "page_num": page_num,
            "text": f"Page {page_num} of {doc_name}",
        })

    logger.info(f"Built {len(chunks)} page-image chunks")
    return chunks


def load_figure_chunks(figures_dir: Path, data_dir: Path) -> List[Dict[str, Any]]:
    """
    Build a chunk list from extracted figures/charts/tables (PNGs).
    Reads figures_metadata.json produced by extract_figures.py.

    image_path is stored RELATIVE to data_dir for portability.

    Returns a list of dicts:
        type        : "figure"
        image_path  : path relative to data_dir (string)
        doc_name    : document name (PDF stem)
        page_num    : int page number
        label       : "picture", "chart", or "table"
        text        : short description used as fallback display text
    """
    figures_dir = Path(figures_dir)
    data_dir = Path(data_dir)
    metadata_path = figures_dir / "figures_metadata.json"

    if not metadata_path.exists():
        logger.warning(f"No figures_metadata.json in {figures_dir}, skipping figures")
        return []

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    chunks = []
    for r in records:
        img_path = figures_dir / r["filename"]
        if not img_path.exists():
            logger.debug(f"Figure image not found: {img_path}")
            continue

        rel_path = img_path.relative_to(data_dir)

        chunks.append({
            "type": "figure",
            "image_path": str(rel_path),
            "doc_name": r["doc_name"],
            "page_num": r["page_num"],
            "label": r["label"],
            "text": f"{r['label']} on page {r['page_num']} of {r['doc_name']}",
        })

    logger.info(f"Built {len(chunks)} figure chunks from {metadata_path.name}")
    return chunks


def load_evidence_chunks(images_dir: Path, data_dir: Path) -> List[Dict[str, Any]]:
    """
    Build a chunk list purely from the nested directory structure.
    
    Expected structure:
    images_dir / {doc_name} / {question_id}_{question_text} / [images...]
    """
    images_dir = Path(images_dir)
    data_dir = Path(data_dir)
    chunks = []

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return chunks

    # Iterate through document directories (e.g., "0b85477387a9d0cc33fca0f4becaa0e5")
    for doc_dir in images_dir.iterdir():
        if not doc_dir.is_dir():
            continue

        doc_name = doc_dir.name

        # Iterate through question directories (e.g., "0169_Who_is_the_commanding...")
        for q_dir in doc_dir.iterdir():
            if not q_dir.is_dir():
                continue

            folder_name = q_dir.name
            
            # Split folder name into question_id and the question text
            # E.g., "0169_Who_is_the..." -> ID: 169, Text: "Who_is_the..."
            parts = folder_name.split("_", 1)
            if len(parts) != 2:
                logger.debug(f"Skipping incorrectly formatted folder: {folder_name}")
                continue
            
            try:
                question_id = int(parts[0])
            except ValueError:
                logger.debug(f"Could not parse question ID from folder: {folder_name}")
                continue
            
            # Reconstruct question text (replace underscores with spaces)
            question = parts[1].replace("_", " ")

            # Grab all image files inside this specific subfolder and create one chunk per image
            for img_file in q_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    # Keep it relative to data_dir for Qdrant portability
                    rel_path = str(img_file.relative_to(data_dir))
                    
                    chunks.append({
                        "type": "evidence",
                        "text": question,
                        "image_path": rel_path,
                        "question_id": question_id,
                        "doc_name": doc_name,
                    })

    logger.info(f"Built {len(chunks)} evidence chunks directly from {images_dir.name}")
    return chunks