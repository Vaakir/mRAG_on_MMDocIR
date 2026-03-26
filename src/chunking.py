"""
Chunking strategies for the mRAG pipeline.

All strategies take a list of unstructured Element blocks and return a list of chunks,
where each chunk is a dict with 'text' and 'metadata'.
"""

from __future__ import annotations
from typing import List, Dict, Any


class Chunker:
    """
    Multiple chunking strategies over unstructured Element blocks.

    Usage:
        from unstructured.partition.auto import partition
        blocks = partition(filename="doc.pdf")
        chunker = Chunker(blocks)

        chunks = chunker.fixed_size(max_chars=1000)
        chunks = chunker.sliding_window(window_chars=1000, overlap_chars=200)
        chunks = chunker.semantic()
        chunks = chunker.hierarchical()
    """

    # Block categories that typically start a new semantic section
    HEADING_CATEGORIES = {"Title", "Header"}

    def __init__(self, blocks: list):
        self.blocks = blocks

    # ------------------------------------------------------------------ #
    #  1. Fixed-size chunking  (your existing strategy, cleaned up)
    # ------------------------------------------------------------------ #
    def fixed_size(self, max_chars: int = 1000) -> List[Dict[str, Any]]:
        """Concatenate blocks until max_chars is exceeded, then start a new chunk."""
        chunks: list[dict] = []
        current_texts: list[str] = []
        current_len = 0

        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            if not text:
                continue
            current_texts.append(text)
            current_len += len(text)

            if current_len >= max_chars or i == len(self.blocks) - 1:
                chunks.append(self._make_chunk(current_texts))
                current_texts = []
                current_len = 0

        return chunks

    # ------------------------------------------------------------------ #
    #  2. Sliding-window chunking  (overlapping fixed-size windows)
    # ------------------------------------------------------------------ #
    def sliding_window(
        self, window_chars: int = 1000, overlap_chars: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Fixed-size windows with overlap so context at chunk boundaries
        is not lost. Good for dense text documents.
        """
        # Flatten blocks into individual text pieces with indices
        pieces: list[tuple[int, str]] = []
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            if text:
                pieces.append((i, text))

        chunks: list[dict] = []
        start = 0

        while start < len(pieces):
            current_texts: list[str] = []
            current_len = 0
            end = start

            while end < len(pieces) and current_len + len(pieces[end][1]) <= window_chars:
                current_texts.append(pieces[end][1])
                current_len += len(pieces[end][1])
                end += 1

            # Always include at least one piece
            if not current_texts and end < len(pieces):
                current_texts.append(pieces[end][1])
                end += 1

            if current_texts:
                chunks.append(self._make_chunk(current_texts))

            # Move start forward by (window - overlap) worth of pieces
            step_chars = 0
            step = start
            target = max(current_len - overlap_chars, 1)
            while step < end and step_chars < target:
                step_chars += len(pieces[step][1])
                step += 1
            start = max(step, start + 1)

        return chunks

    # ------------------------------------------------------------------ #
    #  3. Semantic chunking  (split on headings / category boundaries)
    # ------------------------------------------------------------------ #
    def semantic(self) -> List[Dict[str, Any]]:
        """
        Group blocks into sections separated by Title/Header elements.
        Each heading starts a new chunk. Keeps the document's natural
        semantic boundaries intact.
        """
        chunks: list[dict] = []
        current_texts: list[str] = []

        for block in self.blocks:
            text = block.text.strip()
            if not text:
                continue

            # Start a new chunk when we hit a heading (and have accumulated text)
            if block.category in self.HEADING_CATEGORIES and current_texts:
                chunks.append(self._make_chunk(current_texts))
                current_texts = []

            current_texts.append(text)

        # Flush remaining
        if current_texts:
            chunks.append(self._make_chunk(current_texts))

        return chunks

    # ------------------------------------------------------------------ #
    #  4. Hierarchical chunking  (page → section → paragraph)
    # ------------------------------------------------------------------ #
    def hierarchical(self) -> List[Dict[str, Any]]:
        """
        Two-level hierarchy:
          - Level 1 (parent): full page text  (coarse retrieval)
          - Level 2 (child):  per-section text within each page  (fine retrieval)

        Returns a flat list but each chunk's metadata includes
        'level' ('page' or 'section') and 'page_number'.
        """
        # Group blocks by page number
        pages: dict[int, list] = {}
        for block in self.blocks:
            page = getattr(block.metadata, "page_number", None) or 0
            pages.setdefault(page, []).append(block)

        chunks: list[dict] = []

        for page_num in sorted(pages):
            page_blocks = pages[page_num]
            page_texts = [b.text.strip() for b in page_blocks if b.text.strip()]

            if not page_texts:
                continue

            # Parent chunk: whole page
            chunks.append(
                self._make_chunk(
                    page_texts,
                    extra_meta={"level": "page", "page_number": page_num},
                )
            )

            # Child chunks: split by headings within the page
            section_texts: list[str] = []
            for block in page_blocks:
                text = block.text.strip()
                if not text:
                    continue
                if block.category in self.HEADING_CATEGORIES and section_texts:
                    chunks.append(
                        self._make_chunk(
                            section_texts,
                            extra_meta={"level": "section", "page_number": page_num},
                        )
                    )
                    section_texts = []
                section_texts.append(text)

            if section_texts:
                chunks.append(
                    self._make_chunk(
                        section_texts,
                        extra_meta={"level": "section", "page_number": page_num},
                    )
                )

        return chunks

    # ------------------------------------------------------------------ #
    #  Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_chunk(
        texts: list[str], extra_meta: dict | None = None
    ) -> Dict[str, Any]:
        joined = " ".join(texts)
        meta: dict[str, Any] = {"char_len": len(joined)}
        if extra_meta:
            meta.update(extra_meta)
        return {"text": joined, "metadata": meta}
