# src/preprocessing/build_multimodal_indexes.py
# Standalone index builder — builds one Qdrant collection per chunking strategy.
# Each collection contains text chunks + page images + figures + evidence crops,
# all embedded with Jina CLIP v2 into the shared 1024-D cosine space.
#
# Collections are named:  advanced_fixed_size | advanced_sliding_window |
#                         advanced_semantic   | advanced_hierarchical   |
#                         advanced_enhanced_hierarchical
#
# Usage:
#   cd src
#   python preprocessing/build_multimodal_indexes.py
#   python preprocessing/build_multimodal_indexes.py --force
#   python preprocessing/build_multimodal_indexes.py --strategy semantic

import sys
import logging
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import AdvancedConfig, SRC_DIR
from indexing.embedder import TextEmbedder
from indexing.vector_database import QdrantVectorDB
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from preprocessing.image_processor import (
    load_page_image_chunks,
    load_figure_chunks,
    load_evidence_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES = [
    "fixed_size",
    "sliding_window",
    "semantic",
    "hierarchical",
    "enhanced_hierarchical",
]


def build_index(config: AdvancedConfig):
    """
    Build one multimodal Qdrant collection from the chunk file specified in config.
    Indexes text chunks, page images, extracted figures, and evidence crops.
    Skips rebuild if the collection already exists, unless force_rebuild=True.
    """
    t0 = time.time()

    embedder = TextEmbedder(config.EMBEDDING_MODEL)
    config.USE_MULTIMODAL=True # in preprocessing we want it to build anyway
    vector_db = QdrantVectorDB(config)

    # ── Text chunks ──────────────────────────────────────────────────────────
    chunks_path = Path(config.PREPROCESSED_CHUNKS_FILE)
    logger.info(f"Loading text chunks from {chunks_path.name}")
    chunks = load_preprocessed_chunks(chunks_path)
    print_chunk_statistics(chunks)

    logger.info(f"Embedding {len(chunks)} text chunks...")
    text_embeddings = embedder.embed_texts(
        [c["text"] for c in chunks],
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )

    # ── Page images ───────────────────────────────────────────────────────────
    page_chunks, page_embeddings = [], None
    if config.USE_MULTIMODAL:
        page_chunks = load_page_image_chunks(config.PAGE_IMAGES_TRAIN_DIR, config.DATA_DIR)
        if page_chunks:
            logger.info(f"Embedding {len(page_chunks)} page images...")
            abs_paths = [str(config.DATA_DIR / c["image_path"]) for c in page_chunks]
            page_embeddings = embedder.embed_images(abs_paths)

    # ── Figures ───────────────────────────────────────────────────────────────
    figure_chunks, figure_embeddings = [], None
    if config.USE_MULTIMODAL and hasattr(config, "FIGURES_TRAIN_DIR"):
        figure_chunks = load_figure_chunks(config.FIGURES_TRAIN_DIR, config.DATA_DIR)
        if figure_chunks:
            logger.info(f"Embedding {len(figure_chunks)} figures...")
            abs_paths = [str(config.DATA_DIR / c["image_path"]) for c in figure_chunks]
            figure_embeddings = embedder.embed_images(abs_paths)

    # ── Evidence crops ────────────────────────────────────────────────────────
    evidence_chunks, evidence_embeddings = [], None
    if config.USE_MULTIMODAL and hasattr(config, "IMAGES_TRAIN_DIR"):
        evidence_chunks = load_evidence_chunks(config.IMAGES_TRAIN_DIR, config.DATA_DIR)
        if evidence_chunks:
            logger.info(f"Embedding {len(evidence_chunks)} evidence crops...")
            abs_paths = []
            for c in evidence_chunks:
                img_val = c.get("image_path") or c.get("image_paths")
                abs_paths.append(str(config.DATA_DIR / (img_val[0] if isinstance(img_val, list) else img_val)))
            evidence_embeddings = embedder.embed_images(abs_paths)

    # ── Index into Qdrant ─────────────────────────────────────────────────────
    vector_db.create_collection(force_recreate=True)
    qdrant_docs = []
    doc_id = 0

    for chunk, emb in zip(chunks, text_embeddings):
        qdrant_docs.append({
            "id": doc_id, "embedding": emb, "text": chunk["text"],
            "metadata": {
                "type": "text",
                "pdf_name": str(chunk.get("pdf_name", "unknown")),
                "pdf_path": str(chunk.get("pdf_path", "unknown")),
                "chunk_id": chunk.get("chunk_id"),
                "char_len": int(chunk.get("char_len", 0)),
                "page_numbers": chunk.get("page_numbers"),
            },
        })
        doc_id += 1

    if page_embeddings is not None:
        for chunk, emb in zip(page_chunks, page_embeddings):
            qdrant_docs.append({
                "id": doc_id, "embedding": emb, "text": chunk["text"],
                "metadata": {
                    "type": "page_image",
                    "image_path": chunk["image_path"],
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                },
            })
            doc_id += 1

    if figure_embeddings is not None:
        for chunk, emb in zip(figure_chunks, figure_embeddings):
            qdrant_docs.append({
                "id": doc_id, "embedding": emb, "text": chunk["text"],
                "metadata": {
                    "type": "figure",
                    "image_path": chunk["image_path"],
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                    "label": chunk["label"],
                },
            })
            doc_id += 1

    if evidence_embeddings is not None:
        for chunk, emb in zip(evidence_chunks, evidence_embeddings):
            img_val = chunk.get("image_path") or chunk.get("image_paths")
            qdrant_docs.append({
                "id": doc_id, "embedding": emb, "text": chunk["text"],
                "metadata": {
                    "type": chunk.get("type", "evidence"),
                    "image_path": img_val,
                    "question_id": chunk.get("question_id"),
                    "page_num": chunk.get("page_num"),
                    "doc_name": chunk.get("doc_name"),
                },
            })
            doc_id += 1

    vector_db.index_documents(qdrant_docs)
    logger.info(
        f"Indexed {doc_id} docs  "
        f"({len(chunks)} text | {len(page_chunks)} page images | "
        f"{len(figure_chunks)} figures | {len(evidence_chunks)} evidence)  "
        f"in {time.time() - t0:.1f}s"
    )


def build_for_strategy(strategy: str):
    chunks_file = SRC_DIR / "data" / "preprocessed" / f"chunks_{strategy}.json"
    if not chunks_file.exists():
        logger.error(f"Chunk file not found, skipping: {chunks_file}")
        return

    config = AdvancedConfig()
    config.PREPROCESSED_CHUNKS_FILE = chunks_file
    config.VECTOR_DB_COLLECTION = f"advanced_{strategy}"

    logger.info("=" * 70)
    logger.info(f"Strategy  : {strategy}")
    logger.info(f"Chunks    : {chunks_file.name}")
    logger.info(f"Collection: {config.VECTOR_DB_COLLECTION}")
    logger.info("=" * 70)

    build_index(config)
    logger.info(f"Finished: {strategy}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal Qdrant indexes for all chunking strategies."
    )
    parser.add_argument("--strategy", choices=STRATEGIES, help="Build only this strategy (default: all)")
    args = parser.parse_args()

    targets = [args.strategy] if args.strategy else STRATEGIES
    t0 = time.time()
    for strategy in targets:
        build_for_strategy(strategy)
    logger.info(f"All done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
