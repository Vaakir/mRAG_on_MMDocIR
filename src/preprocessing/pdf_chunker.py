import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Tuple
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import json
from config.config import PREPROCESSED_DOCUMENTS_FILE, PREPROCESSED_DATA_DIR, BaselineConfig
try:
    from .pdf_loader import load_read_documents
except ImportError:
    from pdf_loader import load_read_documents

class Chunking:
    HEADING_CATEGORIES = {"Title", "Header"}

    def __init__(self, blocks: List):
        self.blocks = Chunking.normalize_blocks(blocks)

    @staticmethod
    def normalize_blocks(blocks: List) -> List:
        """
        Normalize blocks before chunking:

        1. Merge consecutive Header/Title blocks into one.
           Rationale: docling sometimes splits a single heading across multiple
           blocks; merging them preserves the full semantic heading.

        2. Attach orphaned NarrativeText (blocks that appear before any header
           has been seen) to the next Header/Title block encountered.
           The combined block keeps the Header category so all chunking methods
           treat it as a section boundary.
           Rationale: text at the top of a document (e.g. release dates, intro
           sentences) has no header context — attaching it to the first header
           gives it one.
        """
        if not blocks:
            return blocks

        # --- Pass 1: merge consecutive headers ---
        merged = []
        for block in blocks:
            text = block.text.strip() if block.text else ""
            if not text:
                continue
            if (
                merged
                and block.category in Chunking.HEADING_CATEGORIES
                and merged[-1].category in Chunking.HEADING_CATEGORIES
            ):
                prev = merged[-1]
                merged[-1] = SimpleNamespace(
                    text=prev.text + " " + text,
                    category=prev.category,
                    page_number=prev.page_number,
                )
            else:
                merged.append(SimpleNamespace(
                    text=text,
                    category=block.category,
                    page_number=block.page_number,
                ))

        # --- Pass 2: attach orphaned NarrativeText to the next header ---
        result = []
        orphan_buffer = []  # NarrativeText blocks seen before any header
        seen_header = False

        for block in merged:
            if block.category in Chunking.HEADING_CATEGORIES:
                seen_header = True
                if orphan_buffer:
                    orphan_text = " ".join(b.text for b in orphan_buffer)
                    result.append(SimpleNamespace(
                        text=orphan_text + " " + block.text,
                        category=block.category,
                        page_number=orphan_buffer[0].page_number,
                    ))
                    orphan_buffer = []
                else:
                    result.append(block)
            else:
                if not seen_header:
                    orphan_buffer.append(block)
                else:
                    result.append(block)

        # If there was never a header at all, just keep the orphaned text as-is
        result.extend(orphan_buffer)

        # --- Pass 3: remove repeated running headers ---
        # Page-level running headers (e.g. "Is the service safe?") repeat every page
        # of a section. Between repetitions there are sub-headers ("The failure to…")
        # that overwrite a naive last_seen check, so we keep a small rolling window
        # of recent header texts and skip any new header that is an exact match or
        # a prefix-match of something in that window.
        _WINDOW = 8
        deduped = []
        recent_headers: list = []  # most-recent at index -1
        for block in result:
            if block.category in Chunking.HEADING_CATEGORIES:
                is_dupe = any(
                    block.text == h or h.startswith(block.text)
                    for h in recent_headers
                )
                if is_dupe:
                    continue
                recent_headers.append(block.text)
                if len(recent_headers) > _WINDOW:
                    recent_headers.pop(0)
            deduped.append(block)

        return deduped

    @staticmethod
    def _merge_small_chunks(
        chunks: List[Dict[str, Any]],
        min_chars: int = 1200,
        max_chars: int = 3000,
    ) -> List[Dict[str, Any]]:
        """
        Forward-merge consecutive chunks that are below min_chars.
        A merge is skipped if it would push the combined text over max_chars.
        Page numbers are unioned across merged chunks.
        """
        result = []
        i = 0
        while i < len(chunks):
            chunk = dict(chunks[i])
            while chunk["char_len"] < min_chars and i + 1 < len(chunks):
                nxt = chunks[i + 1]
                if chunk["char_len"] + 1 + nxt["char_len"] > max_chars:
                    break
                merged_text = chunk["text"] + " " + nxt["text"]
                merged_pages = sorted(
                    set(chunk.get("page_numbers", []) + nxt.get("page_numbers", []))
                )
                chunk = {
                    **chunk,
                    "text": merged_text,
                    "char_len": len(merged_text),
                    "page_numbers": merged_pages,
                }
                i += 1
            result.append(chunk)
            i += 1
        return result

    @staticmethod
    def _split_oversized(chunks: List[Dict[str, Any]], max_chars: int = 3000) -> List[Dict[str, Any]]:
        """
        Split any chunk exceeding max_chars into fixed-size sub-chunks.
        Preserves all metadata from the original chunk.
        """
        result = []
        for chunk in chunks:
            if chunk["char_len"] <= max_chars:
                result.append(chunk)
                continue
            text = chunk["text"]
            meta = {k: v for k, v in chunk.items() if k not in ("text", "char_len")}
            for i in range(0, len(text), max_chars):
                sub_text = text[i:i + max_chars]
                result.append({"text": sub_text, "char_len": len(sub_text), **meta})
        return result

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

            # Split oversized blocks into sub-chunks of max 1700 chars
            if len(text) > max_chars:
                sub_texts = [text[j:j+max_chars] for j in range(0, len(text), max_chars)]
            else:
                sub_texts = [text]

            for sub_text in sub_texts:
                # Check BEFORE adding if the current chunk would exceed limit
                if text_len + len(sub_text) > max_chars and current_texts:
                    # Flush current chunk
                    page_numbers_list = {"page_numbers": list(page_numbers["page_numbers"])}
                    chunks.append(Chunking._make_chunk(current_texts, page_numbers_list))
                    current_texts = []
                    page_numbers = {"page_numbers": set()}
                    text_len = 0
            
                # Now add the sub_text
                current_texts.append(sub_text)
                text_len += len(sub_text)

        if current_texts:
            page_numbers = {"page_numbers": list(page_numbers["page_numbers"])}
            chunks.append(Chunking._make_chunk(current_texts, page_numbers))
        return chunks

    def sliding_window(
        self, max_chars: int = 1000, overlap_chars: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Block-based sliding window.
        Behaves like fixed_size (stacks whole blocks up to max_chars), but 
        carries over previous blocks to satisfy the overlap_chars requirement 
        for the next chunk.
        """
        current_blocks = []  # Stores tuples of (text, page_num)
        chunks = []
        text_len = 0
        
        for i, block in enumerate(self.blocks):
            text = block.text.strip()
            page_num = getattr(block, "page_number", 0)
            
            if not text:
                continue

            # If a single block is massive, slice it into smaller pieces 
            # (the size of overlap_chars) so the overlap logic can grab exact amounts.
            if len(text) > max_chars:
                # E.g., chops the 200k block into 200-character pieces
                sub_texts = [text[j:j+overlap_chars] for j in range(0, len(text), overlap_chars)]
            else:
                sub_texts = [text]

            for sub_text in sub_texts:
                # If adding this block exceeds the limit, flush and calculate overlap
                if text_len + len(sub_text) > max_chars and current_blocks:
                    
                    # 1. Package and flush the current chunk
                    chunk_texts = [b[0] for b in current_blocks]
                    chunk_pages = set(b[1] for b in current_blocks)
                    chunks.append(
                        Chunking._make_chunk(
                            chunk_texts, 
                            extra_meta={"page_numbers": sorted(list(chunk_pages))}
                        )
                    )
                    
                    # 2. Carry over blocks from the end until we hit the overlap target
                    overlap_blocks = []
                    overlap_len = 0
                    
                    # Go backwards through the blocks we just flushed
                    for past_block in reversed(current_blocks):
                        overlap_blocks.insert(0, past_block)
                        overlap_len += len(past_block[0])
                        # Stop once we've grabbed enough text to satisfy the overlap
                        if overlap_len >= overlap_chars:
                            break
                    
                    # 3. Reset state for the next chunk, starting with the overlap
                    current_blocks = overlap_blocks
                    text_len = overlap_len

                # Append the new text
                current_blocks.append((sub_text, page_num))
                text_len += len(sub_text)

        # Flush anything remaining at the end
        if current_blocks:
            chunk_texts = [b[0] for b in current_blocks]
            chunk_pages = set(b[1] for b in current_blocks)
            chunks.append(
                Chunking._make_chunk(
                    chunk_texts, 
                    extra_meta={"page_numbers": sorted(list(chunk_pages))}
                )
            )
            
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

        return self._merge_small_chunks(self._split_oversized(chunks))

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

        return self._split_oversized(hierarchical_pages)

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

        return self._split_oversized(hierarchical_sections)


def chunk_and_save_pdf_data(
    all_documents: List[Dict[str, Any]], output_dir: str = BaselineConfig.PREPROCESSED_DATA_DIR
) -> Dict[str, Any]:
    """
    Applies multiple chunking methods to the documents and saves each method's result to a JSON file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunk_methods = {
        "fixed_size": lambda c: c.fixed_size(),
        "sliding_window": lambda c: c.sliding_window(),
        "semantic": lambda c: c.semantic(),
        "hierarchical": lambda c: c.hierarchical(),
        "enhanced_hierarchical": lambda c: c.enhanced_hierarchical(),
    }

    for method_name, method_fn in chunk_methods.items():
        all_chunked_docs = []
        for doc in all_documents:
            chunker = Chunking(doc["blocks"])

            chunked_doc = {
                **doc,
                "chunks": method_fn(chunker),
            }

            chunked_doc.pop("blocks", None)
            all_chunked_docs.append(chunked_doc)

        # save ONE file per method
        final_path = out_path / f"chunks_{method_name}.json"
        save_chunked_pdf_data(all_chunked_docs, path=str(final_path))

    return {"status": "success", "methods_processed": list(chunk_methods.keys())}


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

    cutoff = np.percentile(lengths, percentile)
    filtered = [l for l in lengths if l <= cutoff]
    plt.hist(filtered, bins=30, label=label, alpha=alpha)


if __name__ == "__main__":
    all_documents = load_read_documents(PREPROCESSED_DOCUMENTS_FILE)
    result = chunk_and_save_pdf_data(all_documents, output_dir=PREPROCESSED_DATA_DIR)
