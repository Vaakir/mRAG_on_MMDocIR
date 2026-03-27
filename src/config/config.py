# src/config/config.py

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src"
PDF_DIR = DATA_DIR / "pdfs_train" / "pdfs_train"
COLLECTION_DIR = DATA_DIR / "project_collection"
TRAIN_JSONL = COLLECTION_DIR / "train" / "train.jsonl"
TEST_JSONL = COLLECTION_DIR / "test" / "test.jsonl"

# Cache directory for processed PDFs (for manual inspection)
CACHE_DIR = PROJECT_ROOT / "cache"

# Pre-processed chunks from team (fixed-size chunking already done)
PREPROCESSED_CHUNKS_FILE = DATA_DIR / "data" / "chunks_fixed_size.json"
USE_PREPROCESSED_CHUNKS = True  # Set to True to use team's chunks

# Model settings
# Using CLIP for embeddings (multimodal - ready for image embeddings later)
EMBEDDING_MODEL = "ViT-B/32"  # CLIP model (text encoder for baseline, can add images later)
EMBEDDING_DIMENSION = 512  # Dimension for CLIP ViT-B/32

# Retrieval settings
TOP_K = 5  # Number of documents to retrieve

# Chunking settings (team's fixed-size strategy for baseline)
CHUNKING_STRATEGY = "fixed_size"  # Options: fixed_size, sliding_window, semantic, hierarchical
CHUNK_SIZE = 1000  # Characters per chunk (as per team's implementation)
CHUNK_OVERLAP = 200  # Overlap between chunks (for sliding_window strategy)

# Vector Database settings (Qdrant - as per team decision)
VECTOR_DB_MODE = "local"  # Options: "local", "memory", "docker"
VECTOR_DB_PATH = str(PROJECT_ROOT / "local_qdrant")  # Local storage path
VECTOR_DB_COLLECTION = "baseline_documents"  # Collection name
VECTOR_DB_DISTANCE = "COSINE"  # Distance metric: COSINE, DOT, MANHATTAN, EUCLID

# LLM settings (using Ollama)
OLLAMA_BASE_URL = "https://ollama.ux.uis.no"  # UiS Ollama cluster
# Using lightweight model for baseline (fast response). Switch to qwen3-vl:8b for multimodal system
LLM_MODEL = "qwen3-vl:8b"  # Alternatives: qwen3-vl:8b (multimodal), qwen3:32b, llama3.1:70b

# Random seed for reproducibility
RANDOM_SEED = 42