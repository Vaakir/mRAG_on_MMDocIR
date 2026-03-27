from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class Chunking:
    HEADING_CATEGORIES = {"Title", "Header"}

    def __init__(self, blocks: List):
        self.blocks = blocks

    @staticmethod
    def _make_chunk(
        texts: List[str], extra_meta: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Helper function to store char_len which can be useful downstream"""
        chunk_text = " ".join(texts)
        chunk = {"text": chunk_text, "char_len": len(chunk_text)}
        if extra_meta:
            chunk.update(extra_meta)
        return chunk

    def fixed_size(self, max_chars: int = 1000) -> List[Dict[str, Any]]:
        """Concatenate blocks until max_chars is exceeded, then start a new chunk."""

        current_texts, chunks = [], []
        text_len, blocks_count = 0, len(self.blocks)
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            if not text:
                continue

            text_len += len(text)
            current_texts.append(text)
            if text_len > max_chars or i == blocks_count - 1:
                chunks.append(Chunking._make_chunk(current_texts))
                current_texts = []
                text_len = 0
        return chunks

    def sliding_window(
        self, window_charts: int = 1000, overlap_chars: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Fixed-size windows with overlap so context at chunk boundaries
        is not lost. Good for dense text documents.
        """
        current_texts, chunks = [], []
        text_len, blocks_count = 0, len(self.blocks)
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            if not text:
                continue

            text_len += len(text)
            current_texts.append(text)
            if text_len > window_charts or i == blocks_count - 1:
                if chunks:
                    prev_text = chunks[-1]["text"][
                        -overlap_chars:
                    ]  # use the actual text from the last chunk
                    current_texts = [prev_text] + current_texts
                else:
                    prev_text = ""

                current_texts.append(prev_text)
                chunks.append(Chunking._make_chunk(current_texts))
                current_texts = []
                text_len = overlap_chars

        return chunks

    def semantic(self) -> List[Dict[str, Any]]:
        """
        Group blocks into sections separated by title/header elements.
        Each heading starts a new chunk. This keeps the documents natural
        semantic boundaries intact.

        Consecutive headers without body text are merged forward so they
        attach to the next chunk that has actual content.
        """

        current_texts, chunks = [], []
        chunk_contains_text = False

        for block in self.blocks:
            text = block.text.strip()
            if not text:
                continue

            if block.category in self.HEADING_CATEGORIES:
                # Only flush if the current chunk has body text
                if chunk_contains_text:
                    chunks.append(Chunking._make_chunk(current_texts))
                    current_texts = []
                    chunk_contains_text = False

                current_texts.append(text)

            else:
                current_texts.append(text)
                chunk_contains_text = True

        # Add remaining - but if it's only headers, merge into the last chunk
        if current_texts:
            if chunk_contains_text or not chunks:
                chunks.append(Chunking._make_chunk(current_texts))
            else:
                # Trailing headers only - append to previous chunk
                prev = chunks[-1]
                combined = prev["text"] + " " + " ".join(current_texts)
                chunks[-1] = {"text": combined, "char_len": len(combined)}

        return chunks

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
                Chunking._make_chunk(
                    page_texts,
                    extra_meta={"level": "page", "page_number": page_num},
                )
            )

            # Child chunks: split by headings within the page
            section_texts: list[str] = []
            has_body = False
            for block in page_blocks:
                text = block.text.strip()
                if not text:
                    continue
                if block.category in self.HEADING_CATEGORIES:
                    if has_body and section_texts:
                        chunks.append(
                            Chunking._make_chunk(
                                section_texts,
                                extra_meta={
                                    "level": "section",
                                    "page_number": page_num,
                                },
                            )
                        )
                        section_texts = []
                        has_body = False
                    section_texts.append(text)
                else:
                    section_texts.append(text)
                    has_body = True

            if section_texts:
                if has_body or not chunks:
                    chunks.append(
                        Chunking._make_chunk(
                            section_texts,
                            extra_meta={"level": "section", "page_number": page_num},
                        )
                    )
                else:
                    # Trailing headers - merge into last section chunk for this page
                    prev = chunks[-1]
                    combined = prev["text"] + " " + " ".join(section_texts)
                    chunks[-1] = {**prev, "text": combined, "char_len": len(combined)}

        return chunks


def plot_chunk_size_distribution(chunks, label="", alpha=0.5, percentile=99.5):
    lengths = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            lengths.append(chunk["char_len"])
        else:
            lengths.append(len(chunk))

    if not lengths:
        return

    # compute cutoff
    cutoff = np.percentile(lengths, percentile)

    # filter outliers
    filtered = [l for l in lengths if l <= cutoff]

    plt.hist(filtered, bins=30, label=label, alpha=alpha)


if __name__ == "__main__":
    # local tests here..
    pass