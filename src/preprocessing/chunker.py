# src/preprocessing/chunker.py
# Using team's chunking strategies (from Chunking.ipynb notebook)

from typing import List, Dict, Any


class Chunking:
    """
    Chunking strategies for text documents.
    Based on team's implementation in Chunking.ipynb
    
    Supports multiple strategies:
    - fixed_size: Fixed character length chunks (BASELINE)
    - sliding_window: Overlapping chunks
    - semantic: Group by document structure (headings)
    - hierarchical: Two-level (page + section)
    """
    
    SECTION_BOUNDARIES = {"Title"}          # only these start new sections
    SKIP_CATEGORIES    = {"Header", "Footer"}  # layout noise — excluded from all chunks

    @staticmethod
    def _is_real_title(text: str) -> bool:
        """Filter out misclassified titles (question fragments, continuations)."""
        t = text.strip()
        if not t or not t[0].isupper():  return False  # doesn't start with capital
        if t.endswith('?'):              return False  # question fragment
        if t.endswith(','):              return False  # mid-sentence continuation
        if t.endswith(';'):              return False  # mid-sentence continuation
        return True

    def __init__(self, blocks: List):
        """
        Initialize chunker with unstructured blocks.
        
        Args:
            blocks: List of Element objects from unstructured.partition
                    Each has .text, .category, .metadata attributes
        """
        self.blocks = blocks

    @staticmethod
    def _make_chunk(texts: List[str], extra_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """Helper function to create chunk dict with char_len"""
        chunk_text = " ".join(texts)
        chunk = {"text": chunk_text, "char_len": len(chunk_text)}
        if extra_meta:
            chunk.update(extra_meta)
        return chunk

    def fixed_size(self, max_chars: int = 1000) -> List[Dict[str, Any]]:
        """
        Concatenate blocks until max_chars is exceeded, then start a new chunk.
        This is the BASELINE strategy as decided by the team.
        """
        current_texts, chunks = [], []
        text_len, blocks_count = 0, len(self.blocks)
        
        for i, block in enumerate(self.blocks):
            if block.category in Chunking.SKIP_CATEGORIES:
                continue
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
        self, window_chars: int = 1000, overlap_chars: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Fixed-size windows with overlap so context at chunk boundaries
        is not lost. Good for dense text documents.
        """
        current_texts, chunks = [], []
        text_len, blocks_count = 0, len(self.blocks)
        
        for i, block in enumerate(self.blocks):
            if block.category in Chunking.SKIP_CATEGORIES:
                continue
            text = block.text.strip()
            if not text:
                continue

            text_len += len(text)
            current_texts.append(text)

            if text_len > window_chars or i == blocks_count - 1:
                if chunks:
                    prev_text = chunks[-1]["text"][-overlap_chars:]
                    current_texts = [prev_text] + current_texts

                chunks.append(Chunking._make_chunk(current_texts))
                current_texts = []
                text_len = overlap_chars

        return chunks

    def semantic(self) -> List[Dict[str, Any]]:
        """
        Group blocks into sections separated by title/header elements.
        Each heading starts a new chunk. This keeps the documents natural
        semantic boundaries intact.
        """
        current_texts, chunks = [], []
        chunk_contains_text = False
        
        for block in self.blocks:
            if block.category in Chunking.SKIP_CATEGORIES:
                continue
            text = block.text.strip()
            if not text:
                continue

            if block.category in Chunking.SECTION_BOUNDARIES and Chunking._is_real_title(block.text):
                if chunk_contains_text:
                    chunks.append(Chunking._make_chunk(current_texts))
                    current_texts = []
                    chunk_contains_text = False
                current_texts.append(text)
            else:
                current_texts.append(text)
                chunk_contains_text = True
        
        if current_texts:
            if chunk_contains_text or not chunks:
                chunks.append(Chunking._make_chunk(current_texts))
            else:
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

            chunks.append(
                Chunking._make_chunk(
                    page_texts,
                    extra_meta={"level": "page", "page_number": page_num},
                )
            )

            section_texts: list[str] = []
            has_body = False
            for block in page_blocks:
                if block.category in Chunking.SKIP_CATEGORIES:
                    continue
                text = block.text.strip()
                if not text:
                    continue
                if block.category in Chunking.SECTION_BOUNDARIES and Chunking._is_real_title(block.text):
                    if has_body and section_texts:
                        chunks.append(
                            Chunking._make_chunk(
                                section_texts,
                                extra_meta={"level": "section", "page_number": page_num},
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
                    prev = chunks[-1]
                    combined = prev["text"] + " " + " ".join(section_texts)
                    chunks[-1] = {**prev, "text": combined, "char_len": len(combined)}

        return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    strategy: str = "fixed_size",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Chunk all documents using specified strategy.
    
    Args:
        documents: List of documents from process_all_pdfs (with 'blocks')
        strategy: "fixed_size" (baseline), "sliding_window", "semantic", "hierarchical"
        **kwargs: Strategy-specific parameters (e.g., max_chars, window_chars)
    
    Returns:
        List of chunks with metadata:
        - text: chunk text
        - char_len: length
        - pdf_name: source PDF
        - pdf_path: source PDF path
        - chunk_id: sequential ID within document
        - (strategy-specific fields)
    """
    all_chunks = []
    
    for doc in documents:
        blocks = doc.get("blocks", [])
        if not blocks:
            continue
        
        chunker = Chunking(blocks)
        
        # Select strategy
        if strategy == "fixed_size":
            max_chars = kwargs.get("max_chars", 1000)
            chunks = chunker.fixed_size(max_chars=max_chars)
        elif strategy == "sliding_window":
            window_chars = kwargs.get("window_chars", 1000)
            overlap_chars = kwargs.get("overlap_chars", 200)
            chunks = chunker.sliding_window(window_chars=window_chars, overlap_chars=overlap_chars)
        elif strategy == "semantic":
            chunks = chunker.semantic()
        elif strategy == "hierarchical":
            chunks = chunker.hierarchical()
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Add document metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["pdf_name"] = doc["pdf_name"]
            chunk["pdf_path"] = doc["pdf_path"]
            chunk["chunk_id"] = i
            all_chunks.append(chunk)
    
    return all_chunks