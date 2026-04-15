# src/add_evidence_to_index.py

import sys
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
from preprocessing.image_processor import load_evidence_chunks
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def get_max_id(vector_db, collection_name):
    """Safely get the maximum numeric ID in the collection."""
    max_id = -1
    offset = None

    while True:
        points, offset = vector_db.client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )

        if not points:
            break

        for p in points:
            if isinstance(p.id, int):
                max_id = max(max_id, p.id)

        if offset is None:
            break

    return max_id


def main():
    config = AdvancedConfig()

    # 1. Connect
    logger.info(f"Connecting to collection: {config.VECTOR_DB_COLLECTION}")
    embedder = TextEmbedder(config.EMBEDDING_MODEL)
    vector_db = QdrantVectorDB(config)

    existing_count = vector_db.count_documents()
    logger.info(f"Existing documents in collection: {existing_count}")

    if existing_count == 0:
        logger.error("Collection is empty — run full pipeline first.")
        sys.exit(1)

    # --------------------------------------------
    # COUNT + DELETE evidence
    # --------------------------------------------
    evidence_filter = Filter(
        must=[
            FieldCondition(
                key="type",
                match=MatchValue(value="evidence")
            )
        ]
    )

    # Count BEFORE
    evidence_before = vector_db.client.count(
        collection_name=config.VECTOR_DB_COLLECTION,
        count_filter=evidence_filter,
        exact=True
    ).count

    logger.info(f"Evidence BEFORE delete: {evidence_before}")

    # Delete
    vector_db.client.delete(
        collection_name=config.VECTOR_DB_COLLECTION,
        points_selector=evidence_filter,
        wait=True
    )

    # Force persistence (important for local Qdrant)
    try:
        vector_db.client.force_flush()
    except Exception:
        pass

    # Count AFTER
    evidence_after = vector_db.client.count(
        collection_name=config.VECTOR_DB_COLLECTION,
        count_filter=evidence_filter,
        exact=True
    ).count

    logger.info(f"Evidence AFTER delete: {evidence_after}")
    logger.info(f"Deleted: {evidence_before - evidence_after}")

    # Safety check
    if evidence_after != 0:
        logger.error("Deletion incomplete. Aborting to prevent duplication.")
        sys.exit(1)

    # --------------------------------------------
    # LOAD evidence
    # --------------------------------------------
    images_dir = config.IMAGES_TRAIN_DIR
    logger.info(f"Loading evidence from: {images_dir}")

    evidence_chunks = load_evidence_chunks(images_dir, config.DATA_DIR)

    if not evidence_chunks:
        logger.error("No evidence chunks found.")
        sys.exit(1)

    logger.info(f"Loaded {len(evidence_chunks)} evidence chunks")

    # --------------------------------------------
    # EMBED images
    # --------------------------------------------
    abs_paths = []
    for c in evidence_chunks:
        img_val = c.get("image_path") or c.get("image_paths")

        if isinstance(img_val, list):
            abs_paths.append(str(config.DATA_DIR / img_val[0]))
        else:
            abs_paths.append(str(config.DATA_DIR / img_val))

    embeddings = embedder.embed_images(
        abs_paths,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )

    # --------------------------------------------
    # SAFE ID GENERATION
    # --------------------------------------------
    max_id = get_max_id(vector_db, config.VECTOR_DB_COLLECTION)
    start_id = max_id + 1

    logger.info(f"Starting new IDs from: {start_id}")

    docs = []
    for i, (chunk, emb) in enumerate(zip(evidence_chunks, embeddings)):
        docs.append({
            "id": start_id + i,
            "embedding": emb,
            "text": chunk["text"],
            "metadata": {
                "type": chunk["type"],
                "image_path": chunk.get("image_path") or chunk.get("image_paths"),
                "question_id": chunk.get("question_id"),
                "doc_name": chunk.get("doc_name")
            },
        })

    # --------------------------------------------
    # INSERT
    # --------------------------------------------
    logger.info(f"Indexing {len(docs)} evidence chunks...")
    vector_db.index_documents(docs)

    # Final count
    new_count = vector_db.count_documents()
    logger.info(f"Done. Collection now has {new_count} documents")


if __name__ == "__main__":
    main()
