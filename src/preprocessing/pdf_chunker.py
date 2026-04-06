from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from .pdf_chunker import load_read_documents


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
        page_numbers = {"page_numbers": set()}
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            page_numbers["page_numbers"].add(block.page_number)
            if not text:
                continue

            text_len += len(text)
            current_texts.append(text)
            if text_len > max_chars or i == blocks_count - 1:
                page_numbers = {"page_numbers": list(page_numbers["page_numbers"])}
                chunks.append(Chunking._make_chunk(current_texts, page_numbers))
                current_texts = []
                page_numbers = {"page_numbers": set()}
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
        page_numbers = {"page_numbers": set()}
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            page_numbers["page_numbers"].add(block.page_number)
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
                page_numbers = {"page_numbers": list(page_numbers["page_numbers"])}
                current_texts.append(prev_text)
                chunks.append(Chunking._make_chunk(current_texts, page_numbers))
                current_texts = []
                page_numbers = {"page_numbers": set()}
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
        page_numbers = {"page_numbers": set()}
        for block in self.blocks:
            text = block.text.strip()
            page_numbers["page_numbers"].add(block.page_number)
            if not text:
                continue

            if block.category in self.HEADING_CATEGORIES:
                # Only flush if the current chunk has body text
                if chunk_contains_text:
                    page_numbers = {"page_numbers": list(page_numbers["page_numbers"])}
                    chunks.append(Chunking._make_chunk(current_texts, page_numbers))
                    current_texts = []
                    chunk_contains_text = False
                    page_numbers = {"page_numbers": set()}

                current_texts.append(text)

            else:
                current_texts.append(text)
                chunk_contains_text = True

        # Add remaining - but if it's only headers, merge into the last chunk
        if current_texts:
            if chunk_contains_text or not chunks:
                page_numbers = {"page_numbers": list(page_numbers["page_numbers"])}
                chunks.append(Chunking._make_chunk(current_texts, page_numbers))
                page_numbers = {"page_numbers": set()}
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
            page = getattr(block, "page_number", None) or 0
            pages.setdefault(page, []).append(block)

        hierarchical_pages: list[dict] = []

        for page_num in sorted(pages):
            page_blocks = pages[page_num]
            page_texts = [b.text.strip() for b in page_blocks if b.text.strip()]

            if not page_texts:
                continue

            # Parent chunk: whole page
            page_chunk = Chunking._make_chunk(
                page_texts,
                extra_meta={"level": "page", "page_numbers": [page_num]},
            )

            # Child chunks: split by headings within the page
            page_sections: list[dict] = []
            section_texts: list[str] = []
            has_body = False
            for block in page_blocks:
                text = block.text.strip()
                if not text:
                    continue
                if block.category in self.HEADING_CATEGORIES:
                    if has_body and section_texts:
                        page_sections.append(
                            Chunking._make_chunk(
                                section_texts,
                                extra_meta={
                                    "level": "section",
                                    "page_numbers": [page_num],
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
                if has_body or not page_sections:
                    page_sections.append(
                        Chunking._make_chunk(
                            section_texts,
                            extra_meta={"level": "section", "page_numbers": [page_num]},
                        )
                    )
                else:
                    # Trailing headers - merge into last section chunk for this page
                    prev = page_sections[-1]
                    combined = prev["text"] + " " + " ".join(section_texts)
                    page_sections[-1] = {
                        **prev,
                        "text": combined,
                        "char_len": len(combined),
                    }

            # 3. Nest children inside the parent and append to final list
            page_chunk["chunks"] = page_sections
            hierarchical_pages.append(page_chunk)

        return hierarchical_pages

    def enhanced_hierarchical(self) -> List[Dict[str, Any]]:
        """
        Optimized Two-level hierarchy (Cross-Page):
        - Level 1 (parent): Full Section (Starts at a Heading, ends at the next Heading)
        - Level 2 (child): Individual paragraphs/blocks within that section.

        This prevents context loss when a logical section spills across a physical page break.
        Consecutive headers without body text are merged together.
        """
        hierarchical_sections: List[Dict[str, Any]] = []

        # State tracking for the current section
        current_parent_title = (
            "Document Start"  # Default if no header exists at the very beginning
        )
        current_children: List[Dict[str, Any]] = []
        current_parent_texts: List[str] = []
        current_pages = set()  # Using a set to track all pages this section spans
        has_body = False  # Tracks if the current section has actual content

        for block in self.blocks:
            text = block.text.strip()
            if not text:
                continue

            # Get page number safely, default to 0 if missing
            page_num = getattr(block, "page_number", None) or 0
            is_heading = block.category in getattr(self, "HEADING_CATEGORIES", [])

            if is_heading:
                # 1. Flush ONLY if the current section actually has body text
                if has_body:
                    parent_chunk = self._make_chunk(
                        current_parent_texts,
                        extra_meta={
                            "level": "section",
                            "title": current_parent_title,
                            "page_numbers": sorted(list(current_pages)),
                        },
                    )
                    parent_chunk["chunks"] = current_children
                    hierarchical_sections.append(parent_chunk)

                    # Reset the state for the new section
                    current_parent_title = text
                    current_children = []
                    current_parent_texts = [text]
                    current_pages = {page_num}
                    has_body = False
                else:
                    # 2. Consecutive headings: Merge them together
                    if (
                        current_parent_title == "Document Start"
                        and not current_parent_texts
                    ):
                        current_parent_title = text
                    else:
                        current_parent_title += f" {text}"

                    current_parent_texts.append(text)
                    current_pages.add(page_num)

            else:
                # 3. We are inside a section body. Append text and track the page.
                has_body = True
                current_parent_texts.append(text)
                current_pages.add(page_num)

                # If this is the first child in the section, prepend the title to its text
                child_text = text
                if (
                    len(current_children) == 0
                    and current_parent_title != "Document Start"
                ):
                    child_text = f"{current_parent_title}\n{text}"

                # Create a child chunk for this specific paragraph/block
                current_children.append(
                    self._make_chunk(
                        [child_text],
                        extra_meta={
                            "level": "paragraph",
                            "page_numbers": [page_num],
                            "parent_title": current_parent_title,
                        },
                    )
                )

        # 4. Flush the final section after the loop finishes
        if current_children or current_parent_texts:
            parent_chunk = self._make_chunk(
                current_parent_texts,
                extra_meta={
                    "level": "section",
                    "title": current_parent_title,
                    "page_numbers": sorted(list(current_pages)),
                },
            )
            parent_chunk["chunks"] = current_children
            hierarchical_sections.append(parent_chunk)

        return hierarchical_sections


def save_chunked_pdf_data(all_chunked_docs, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_chunked_docs, f, ensure_ascii=False, indent=2)


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
    all_documents = load_read_documents("../data/all_documents.json")

    chunk_methods = {
        "fixed_size": lambda c: c.fixed_size(),
        "sliding_window": lambda c: c.sliding_window(),
        "semantic": lambda c: c.semantic(),
        "hierarchical": lambda c: c.hierarchical(),
        "enhanced_hierarchical": lambda c: c.enhanced_hierarchical(),
    }

    for method_name, method_fn in chunk_methods.items():
        all_chunked_docs = []  # reset per method

        for doc in all_documents:
            chunker = Chunking(doc["blocks"])

            chunked_doc = {
                **doc,
                "chunks": method_fn(chunker),
            }

            chunked_doc.pop("blocks", None)
            all_chunked_docs.append(chunked_doc)

        # save ONE file per method
        save_chunked_pdf_data(
            all_chunked_docs, path=f"../data/chunks_{method_name}.json"
        )
