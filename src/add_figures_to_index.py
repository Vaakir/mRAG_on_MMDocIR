# src/add_figures_to_index.py
# Adds extracted figures to the existing Qdrant collection without rebuilding.
# Run: python src/add_figures_to_index.py

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure src is on the path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config.config import AdvancedConfig
from indexing.embedder import TextEmbedder
from indexing.vector_database import QdrantVectorDB
from preprocessing.image_processor import load_figure_chunks


def main():
    config = AdvancedConfig()

    # 1. Connect to existing collection
    logger.info(f"Connecting to collection: {config.VECTOR_DB_COLLECTION}")
    embedder = TextEmbedder(config.EMBEDDING_MODEL)
    vector_db = QdrantVectorDB(config)

    existing_count = vector_db.count_documents()
    logger.info(f"Existing documents in collection: {existing_count}")

    if existing_count == 0:
        logger.error("Collection is empty — run the full pipeline first to build the base index.")
        sys.exit(1)

    # 2. Load figure chunks
    figures_dir = config.FIGURES_TRAIN_DIR
    logger.info(f"Loading figures from: {figures_dir}")
    figure_chunks = load_figure_chunks(figures_dir, config.DATA_DIR)

    if not figure_chunks:
        logger.error("No figure chunks found. Run extract_figures.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(figure_chunks)} figure chunks")

    # 3. Embed with CLIP
    abs_paths = [str(config.DATA_DIR / c["image_path"]) for c in figure_chunks]
    figure_embeddings = embedder.embed_images(abs_paths, batch_size=config.EMBEDDING_BATCH_SIZE)

    # 4. Build docs starting after the last existing ID
    start_id = existing_count
    docs = []
    for i, (chunk, emb) in enumerate(zip(figure_chunks, figure_embeddings)):
        docs.append({
            "id": start_id + i,
            "embedding": emb,
            "text": chunk["text"],
            "metadata": {
                "type": "figure",
                "image_path": chunk["image_path"],
                "doc_name": chunk["doc_name"],
                "page_num": chunk["page_num"],
                "label": chunk["label"],
            },
        })

    # 5. Upsert into existing collection
    logger.info(f"Indexing {len(docs)} figures (IDs {start_id} to {start_id + len(docs) - 1})...")
    vector_db.index_documents(docs)

    new_count = vector_db.count_documents()
    logger.info(f"Done. Collection now has {new_count} documents (+{new_count - existing_count} figures)")


if __name__ == "__main__":
    main()
