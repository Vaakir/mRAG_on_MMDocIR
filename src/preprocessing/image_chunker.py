# src/preprocessing/image_chunker.py
# Image chunking strategies for multimodal RAG.
#
# Three strategies:
#   1. page_level     — one chunk per page image (simplest, zero processing)
#   2. sliding_window — groups of N adjacent pages with overlap
#
# Each chunk follows the same dict schema so the rest of the pipeline
# (embedder, Qdrant indexer, retriever) can treat image chunks uniformly.
#
# Chunk schema:
#   {
#       "pdf_name":    str,   # e.g. "DSA-278777.pdf"
#       "page_num":    int,   # primary page (first page for multi-page chunks)
#       "page_nums":   list,  # all pages covered by this chunk
#       "image_paths": list,  # absolute paths to the JPG(s)
#       "chunk_type":  str,   # "image_page" | "image_window"
#       "chunk_id":    str,   # unique identifier
#   }

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def page_level(image_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strategy 1 — Page-level (simplest).
    One image chunk per page. Uses the existing page_images directly.

    Args:
        image_records: Output of multimodal_loader.load_page_images()

    Returns:
        List of image chunks, one per page.
    """
    chunks = []
    for rec in image_records:
        chunks.append({
            "pdf_name":    rec["pdf_name"],
            "page_num":    rec["page_num"],
            "page_nums":   [rec["page_num"]],
            "image_paths": [rec["image_path"]],
            "chunk_type":  "image_page",
            "chunk_id":    rec["chunk_id"],
        })
    logger.info(f"page_level: created {len(chunks)} image chunks")
    return chunks


def sliding_window(
    grouped_images: Dict[str, List[Dict[str, Any]]],
    window: int = 2,
    overlap: int = 1,
) -> List[Dict[str, Any]]:
    """
    Strategy 2 — Sliding window over pages.
    Groups N consecutive pages into one chunk, stepping by (window - overlap).

    Analogy to text sliding window:
        window  = chunk size (number of pages)
        overlap = how many pages are shared between consecutive chunks

    Args:
        grouped_images: Output of multimodal_loader.group_images_by_pdf()
        window:         Number of pages per chunk (default 2)
        overlap:        Number of overlapping pages between chunks (default 1)

    Returns:
        List of image chunks, each covering window pages.
    """
    if overlap >= window:
        raise ValueError(f"overlap ({overlap}) must be less than window ({window})")

    step = window - overlap
    chunks = []

    for pdf_name, pages in grouped_images.items():
        n = len(pages)
        for start in range(0, n, step):
            end = min(start + window, n)
            window_pages = pages[start:end]

            page_nums = [p["page_num"] for p in window_pages]
            image_paths = [p["image_path"] for p in window_pages]
            chunk_id = f"{pdf_name.replace('.pdf', '')}_w{page_nums[0]}-{page_nums[-1]}"

            chunks.append({
                "pdf_name":    pdf_name,
                "page_num":    page_nums[0],
                "page_nums":   page_nums,
                "image_paths": image_paths,
                "chunk_type":  "image_window",
                "chunk_id":    chunk_id,
            })

            # Stop if we've covered all pages
            if end == n:
                break

    logger.info(
        f"sliding_window (window={window}, overlap={overlap}): "
        f"created {len(chunks)} image chunks"
    )
    return chunks
