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


def load_evidence_chunks(images_dir: Path, jsonl_path: Path, data_dir: Path) -> List[Dict[str, Any]]:
    """
    Build a chunk list from labeled evidence crops in the JSONL.

    Each evidence entry embeds the *question text* (so similar test queries
    retrieve it) but stores the image path in metadata so the VLM receives
    the image at generation time.

    image_path is stored relative to data_dir for portability.

    Returns a list of dicts:
        type          : "evidence"
        text          : question text  (what gets embedded)
        image_path    : list of relative path strings
        question_id   : int
        doc_name      : str
    """
    images_dir = Path(images_dir)
    jsonl_path = Path(jsonl_path)
    data_dir = Path(data_dir)
    chunks = []
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            question_id = entry.get("question_id")
            question = entry.get("question", "").strip()
            evidence_images = entry.get("evidence_images", [])

            if not question or not evidence_images:
                continue

            image_paths = []
            for rel_path in evidence_images:
                resolved = _resolve_evidence_path(images_dir, rel_path)
                if resolved.exists():
                    image_paths.append(str(resolved.relative_to(data_dir)))  # relative
                else:
                    logger.debug(f"Evidence image not found: {resolved}")
                    missing += 1

            if not image_paths:
                continue

            pdf_path = entry.get("pdf_path", "")
            doc_name = Path(pdf_path).stem if pdf_path else ""

            chunks.append({
                "type": "evidence",
                "text": question,
                "image_path": image_paths,  # relative paths — portable
                "question_id": question_id,
                "doc_name": doc_name,
            })

    if missing:
        logger.warning(f"{missing} evidence image paths could not be resolved")
    logger.info(f"Built {len(chunks)} evidence chunks from {jsonl_path.name}")
    return chunks
