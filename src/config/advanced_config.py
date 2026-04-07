# src/config/advanced_config.py
# Configuration for the advanced RAG pipeline with query techniques

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AdvancedConfig:
    """
    Configuration for the Advanced RAG Pipeline.
    
    Allows easy customization of:
    - Embedding model and settings
    - LLM model and API settings
    - Vector database configuration
    - Retrieval method (hybrid vs dense-only)
    - Query technique selection
    - Chunking and indexing settings
    - Evaluation settings
    """
    
    # ===== EMBEDDING SETTINGS =====
    EMBEDDING_MODEL: str = "jinaai/jina-clip-v2" #"BAAI/bge-large-en-v1.5"
    """Embedding model name from Hugging Face"""
    
    EMBEDDING_DIMENSION: int = 1024
    """Output dimension of embedding model"""
    
    EMBEDDING_BATCH_SIZE: int = 64
    """Batch size for creating embeddings"""
    
    # ===== LLM / GENERATOR SETTINGS =====
    LLM_MODEL: str = "qwen3:32b"
    """LLM model to use (via Ollama)"""
    
    OLLAMA_BASE_URL: str = "https://ollama.ux.uis.no"
    """Base URL for Ollama API"""
    
    LLM_TEMPERATURE: float = 0.0
    """Temperature for LLM generation (0.0 = deterministic)"""
    
    LLM_TOP_P: float = 0.1
    """Top-P for LLM generation"""
    
    # ===== VECTOR DATABASE SETTINGS =====
    VECTOR_DB_MODE: str = "local"
    """Vector DB mode: 'local', 'memory', or 'docker'"""
    
    VECTOR_DB_PATH: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent / "local_qdrant"))
    """Path for local Qdrant database"""
    
    VECTOR_DB_COLLECTION: str = "baseline_documents_jina"
    """Collection name in Qdrant"""
    
    VECTOR_DB_DISTANCE: str = "COSINE"
    """Distance metric: 'COSINE', 'DOT', 'MANHATTAN', 'EUCLID'"""
    
    # ===== RETRIEVAL SETTINGS =====
    TOP_K: int = 5
    """Number of documents to retrieve"""
    
    USE_HYBRID_RETRIEVAL: bool = True
    """Use hybrid BM25 + dense retrieval, or dense-only"""
    
    # ===== CHUNKING SETTINGS =====
    USE_PREPROCESSED_CHUNKS: bool = True
    """Use pre-processed chunks or process from scratch"""
    
    PREPROCESSED_CHUNKS_FILE: str = "./src/data/chunks_fixed_size.json"
    """Path to pre-processed chunks file (relative to project root)"""
    
    CHUNKING_STRATEGY: str = "fixed_size"
    """Chunking strategy: 'fixed_size', 'sliding_window', 'semantic', 'hierarchical'"""
    
    CHUNK_SIZE: int = 1000
    """Max characters per chunk"""
    
    # ===== QUERY TECHNIQUE SETTINGS =====
    QUERY_TECHNIQUE: str = "standard"
    """
    Query technique to use:
    - 'standard': Basic retrieval
    - 'multi_query': Generate paraphrases
    - 'rag_fusion': Multi-query with RRF scoring
    - 'step_back': Generate simpler question
    - 'hyde': Hypothetical documents
    - 'query_decomposition': Break into sub-questions
    - 'query_rewriting': Improve question formulation
    - 'query_expansion': Add synonyms and related terms
    """
    
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'num_variants': 3,  # Used by multi_query, hyde, query_decomposition, query_expansion
    })
    """Configuration dict for the selected query technique"""
    
    # ===== MULTIMODAL SETTINGS =====
    USE_MULTIMODAL_RETRIEVAL: bool = True
    """Enable image retrieval alongside text retrieval"""

    IMAGE_CHUNKING_STRATEGY: str = "page_level"
    """Image chunking strategy: 'page_level' or 'sliding_window'"""

    IMAGE_SLIDING_WINDOW_SIZE: int = 2
    """Number of pages per sliding window image chunk"""

    IMAGE_SLIDING_WINDOW_OVERLAP: int = 1
    """Overlap in pages between sliding window image chunks"""

    IMAGE_COLLECTION: str = "image_documents_jina"
    """Qdrant collection name for image embeddings"""

    PAGE_IMAGES_DIR: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "project_collection" / "train" / "page_images_train"))
    """Path to page_images_{split}/ directory (set at runtime)"""

    IMAGE_TOP_K: int = 3
    """Number of image chunks to retrieve per query"""

    # ===== EVALUATION SETTINGS =====
    EVAL_SUBSET_SIZE: int = 20
    """Number of test questions to evaluate on"""
    
    RETRIEVAL_WORKERS: int = 4
    """Number of parallel workers for document retrieval (Phase 1)"""
    
    GENERATION_WORKERS: int = 2
    """Number of parallel workers for answer generation (Phase 2)"""
    
    # ===== PATHS =====
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    """Root directory of the project"""
    
    PDF_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "pdfs")
    """Directory containing PDF files"""
    
    CACHE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / ".cache")
    """Cache directory for processed PDFs"""
    
    TRAIN_JSONL: Path = field(default_factory=lambda: Path(__file__).parent.parent / "project_collection" / "train" / "train.jsonl")
    """Path to training data JSONL file"""
    
    def __post_init__(self):
        """Validate paths after initialization."""
        # Convert string paths to Path objects if needed
        if isinstance(self.VECTOR_DB_PATH, str):
            self.VECTOR_DB_PATH = str(Path(self.VECTOR_DB_PATH))
        if isinstance(self.PREPROCESSED_CHUNKS_FILE, str):
            self.PREPROCESSED_CHUNKS_FILE = str(Path(self.PREPROCESSED_CHUNKS_FILE))
    
    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]) -> 'AdvancedConfig':
        """
        Create a config from a dictionary (useful for loading from files).
        
        Args:
            config_dict: Dictionary with config keys and values
            
        Returns:
            AdvancedConfig instance
        """
        # Filter to only include fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


# Preset configurations for quick switching

class BaselineConfig(AdvancedConfig):
    """Configuration matching the baseline pipeline."""
    pass


class FastEmbeddingConfig(AdvancedConfig):
    """Fast embedding using smaller model."""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 128


class MultiQueryConfig(AdvancedConfig):
    """Multi-query technique configuration."""
    QUERY_TECHNIQUE: str = "multi_query"
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'num_variants': 3,
    })


class RAGFusionConfig(AdvancedConfig):
    """RAG-Fusion configuration."""
    QUERY_TECHNIQUE: str = "rag_fusion"
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'num_variants': 3,
    })


class HyDEConfig(AdvancedConfig):
    """HyDE (Hypothetical Documents) configuration."""
    QUERY_TECHNIQUE: str = "hyde"
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'num_variants': 3,
    })
