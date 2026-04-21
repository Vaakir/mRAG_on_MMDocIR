# src/preprocessing/image_processor.py
# Builds chunk lists for page images and evidence crops.
# Page images  → embedded with CLIP image encoder (visual content)
# Evidence crops → embedded with question text (question is the retrieval key,
#                  image path is what gets passed to the VLM at generation time)
#
# Paths stored in Qdrant are RELATIVE to DATA_DIR so the same index works
# on any machine (local Mac, Colab, etc.).  Callers resolve them at runtime.

import base64
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# JSONL evidence_images paths contain a "crops/" segment that does not exist on
# disk — files sit directly in the question folder.
_CROPS_RE = re.compile(r"/crops/")
_PAGE_RE = re.compile(r"__page_(\d+)__")


def resize_image(image_path: str, max_width: int = 1120, max_height: int = 1120, quality: int = 85) -> bytes:
    """
    Read an image, resize to fit within max_width x max_height (preserving aspect ratio),
    and return JPEG bytes.
    
    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels (0 to disable)
        max_height: Maximum height in pixels (0 to disable)
        quality: JPEG compression quality (0-100)
    
    Returns:
        JPEG bytes of the resized image
    """
    from PIL import Image
    import io
    
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Skip resizing if disabled or image is already smaller
    if max_width > 0 or max_height > 0:
        if max_width <= 0:
            max_width = w
        if max_height <= 0:
            max_height = h
        
        if w > max_width or h > max_height:
            # Scale proportionally to fit within bounds
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Encode to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@lru_cache(maxsize=256)
def encode_image(image_path: str, max_width: int = None, max_height: int = None, quality: int = None) -> str:
    """
    Resize and encode an image as base64-encoded JPEG.
    
    If max_width, max_height, or quality are None, loads defaults from config.
    
    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels (None to load from config)
        max_height: Maximum height in pixels (None to load from config)
        quality: JPEG compression quality 0-100 (None to load from config)
    
    Returns:
        Base64-encoded string of the resized JPEG image
    """
    # Load config values if not provided
    if max_width is None or max_height is None or quality is None:
        from config.config import AdvancedConfig
        config = AdvancedConfig()
        max_width = max_width or getattr(config, 'MAX_IMAGE_WIDTH', 1120)
        max_height = max_height or getattr(config, 'MAX_IMAGE_HEIGHT', 1120)
        quality = quality or getattr(config, 'IMAGE_RESIZE_QUALITY', 85)
    
    jpeg_bytes = resize_image(image_path, max_width, max_height, quality)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


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
                    
                    page_match = _PAGE_RE.search(img_file.stem)
                    chunks.append({
                        "type": "evidence",
                        "text": question,
                        "image_path": rel_path,
                        "question_id": question_id,
                        "page_num": int(page_match.group(1)) if page_match else None,
                        "doc_name": doc_name,
                    })

    logger.info(f"Built {len(chunks)} evidence chunks directly from {images_dir.name}")
    return chunks