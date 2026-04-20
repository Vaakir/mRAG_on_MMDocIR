# src/config/config.py
# Configuration for the all systems to share a single source of truth
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file for API keys
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Define paths outside the dataclass for cleaner referencing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env", override=True)

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
RESULTS_DIR = SRC_DIR / "results"
PREPROCESSING_TIME_CSV = RESULTS_DIR / "time_preprocessing.csv"
BASELINE_TIME_CSV = RESULTS_DIR / "time_baseline.csv"
RESULTS_CSV = RESULTS_DIR / "results_baseline.csv"

PDFS_DIR = DATA_DIR / "train" / "pdfs_train"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_DOCUMENTS_FILE = DATA_DIR / "preprocessed" / "all_documents.json"

TRAIN_JSONL = DATA_DIR / "train" / "train.jsonl"
TEST_JSONL = DATA_DIR / "test" / "test.jsonl"

PAGE_IMAGES_TRAIN_DIR = DATA_DIR / "train" / "page_images_train"
IMAGES_TRAIN_DIR = DATA_DIR / "train" / "images_train"
PAGE_IMAGES_TEST_DIR = DATA_DIR / "test" / "page_images_test"
IMAGES_TEST_DIR = DATA_DIR / "test" / "images_test"

CACHE_DIR = SRC_DIR / "cache"
CACHE_DB_PATH = CACHE_DIR / "query_cache.db"
PREPROCESSED_CHUNKS_FILE = SRC_DIR / "data" / "preprocessed" / "chunks_fixed_size.json"



@dataclass
class BaselineConfig:
    """Configuration matching the baseline pipeline."""
    HF_TOKEN: str = os.getenv("HF_TOKEN")

    # ===== PATHS =====
    PROJECT_ROOT: Path = PROJECT_ROOT
    SRC_DIR: Path = SRC_DIR
    DATA_DIR: Path = DATA_DIR

    PDFS_DIR: Path = PDFS_DIR
    PREPROCESSED_DATA_DIR: Path = PREPROCESSED_DATA_DIR
    PREPROCESSED_DOCUMENTS_FILE: str = str(PREPROCESSED_DOCUMENTS_FILE)

    TRAIN_JSONL: Path = TRAIN_JSONL
    TEST_JSONL: Path = TEST_JSONL

    CACHE_DIR: Path = CACHE_DIR
    CACHE_DB_PATH: Path = CACHE_DB_PATH
    RESULTS_CSV: Path = RESULTS_CSV
    PREPROCESSING_TIME_CSV: Path = PREPROCESSING_TIME_CSV
    BASELINE_TIME_CSV: Path = BASELINE_TIME_CSV

    PREPROCESSED_CHUNKS_FILE: str = str(PREPROCESSED_CHUNKS_FILE)

    # ===== EMBEDDING SETTINGS =====
    EMBEDDING_MODEL: str = "jinaai/jina-clip-v2"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 64

    # ===== LLM / GENERATOR SETTINGS =====
    LLM_MODEL: str = "qwen3-vl:8b-instruct"
    # AGENT_LLM_MODEL: str = "qwen3:32b"  # Lightweight LLM for agent decisions (Query Rewriter, Grader, Generator strategy)
    OLLAMA_BASE_URL: str = "https://ollama.ux.uis.no"
    OLLAMA_API_KEY: str = os.getenv('OLLAMA_API_KEY', '')
    """API key for Ollama authentication (loaded from .env, empty string if not found)"""
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 4
    LLM_RETRY_DELAY: int = 30

    # ===== VECTOR DATABASE SETTINGS =====
    VECTOR_DB_MODE: str = "local"
    VECTOR_DB_PATH: str = str(PROJECT_ROOT / "local_qdrant")
    VECTOR_DB_COLLECTION: str = "baseline_documents_jina"
    VECTOR_DB_DISTANCE: str = "COSINE"

    # ===== RETRIEVAL SETTINGS =====
    TOP_K: int = 5
    USE_HYBRID_RETRIEVAL: bool = True

    # ===== CHUNKING SETTINGS =====
    USE_PREPROCESSED_CHUNKS: bool = True
    CHUNKING_STRATEGY: str = "semantic"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CONTEXT_WINDOW: int = 0  # adjacent chunks to prepend/append at retrieval time

    # ===== EVALUATION SETTINGS =====
    EVAL_SUBSET_SIZE: int = 20
    RETRIEVAL_WORKERS: int = 4
    GENERATION_WORKERS: int = 2

    RANDOM_SEED: int = 42

    def __post_init__(self):
        """Validate paths after initialization."""
        if isinstance(self.VECTOR_DB_PATH, Path):
            self.VECTOR_DB_PATH = str(self.VECTOR_DB_PATH)
        if isinstance(self.PREPROCESSED_CHUNKS_FILE, str):
            self.PREPROCESSED_CHUNKS_FILE = Path(self.PREPROCESSED_CHUNKS_FILE)

    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]):
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


@dataclass
class AdvancedConfig(BaselineConfig):
    """Configuration for the Advanced RAG Pipeline."""
    # ===== ADVANCED APP OVERRIDES =====
    # override the chunk file to use the semantic chunks instead of the fixed-size ones
    PREPROCESSED_CHUNKS_FILE: str = str(SRC_DIR / "data" / "preprocessed" / "chunks_semantic.json")

    VECTOR_DB_COLLECTION: str = "advanced_multimodal"
    """Separate collection from baseline so the two don't interfere"""

    # Use one model for everything — no server model swapping = no OOM crashes
    LLM_MODEL: str = "qwen3-vl:8b-instruct"

    # ===== MULTIMODAL SETTINGS =====
    USE_MULTIMODAL: bool = True

    MAX_VLM_IMAGES: int = 2
    """Cap images sent to VLM to avoid OOM"""

    VLM_MODEL: str = "qwen3-vl:8b-instruct"
    """Vision-language model used when image chunks are retrieved"""

    # Sequential — keeps logs readable and avoids concurrent calls on shared GPU
    GENERATION_WORKERS: int = 1
    RETRIEVAL_WORKERS: int = 1

    # ===== IMAGE RESIZING SETTINGS =====
    MAX_IMAGE_WIDTH: int = 1024
    """Maximum width in pixels for image resizing (0 to disable resizing)"""
    MAX_IMAGE_HEIGHT: int = 1024
    """Maximum height in pixels for image resizing (0 to disable resizing)"""
    IMAGE_RESIZE_QUALITY: int = 85
    """JPEG compression quality (0-100, higher = better quality but larger file size)"""
    VLM_USE_RAW_CHATML: bool = True
    """Use raw ChatML with /no_think to bypass qwen3-vl:8b thinking bug (ignores think=false in API)"""

    PAGE_IMAGES_TRAIN_DIR: Path = PAGE_IMAGES_TRAIN_DIR
    IMAGES_TRAIN_DIR: Path = IMAGES_TRAIN_DIR
    PAGE_IMAGES_TEST_DIR: Path = PAGE_IMAGES_TEST_DIR
    IMAGES_TEST_DIR: Path = IMAGES_TEST_DIR

    FIGURES_TRAIN_DIR: Path = DATA_DIR / "train" / "figures_train"

    # ===== RETRIEVAL FILTER =====
    ALLOWED_CHUNK_TYPES: List[str] = field(default_factory=lambda: ["text", "page_image", "figure", "evidence"])

    # ===== QUERY TECHNIQUE SETTINGS =====
    QUERY_TECHNIQUE: str = "standard"
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_variants": 3,
            "max_page_images": 1,
        }
    )

    # ===== PROMPTING STRATEGY SETTINGS =====
    PROMPTING_STRATEGY: str = "standard"
    """
    Prompting strategy for answer generation:
    - 'standard': Direct extraction without special prompting
    - 'few_shot': Provide multiple examples (2-5), then ask question
    - 'role': Assign expert role to LLM (financial_analyst, researcher, etc.)
    - 'cot': Chain-of-Thought - explicit step-by-step reasoning
    - 'ensemble': Multiple strategies with voting/consensus
    """
    
    PROMPTING_STRATEGY_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        # Role strategy
        'role_type': 'rag_specialist',
        
        # CoT strategy
        'show_reasoning': False,  # set to False to hide reasoning
        
        # Ensemble strategy
        'mode': 'multi_prompt',  # 'multi_prompt' or 'self_consistency'
        'ensemble_size': 3,
        'aggregation_method': 'embedding_similarity',  # 'judge', 'combine', 'embedding_similarity'
        'strategies': ['standard', 'cot', 'few_shot', 'financial_analyst_role'],
        'include_strategy_metadata': False,
        'verbose_logging': False,  # Enable detailed ensemble logging
        'temperatures': {
            'standard': 0.5,
            'cot': 0.6,
            'few_shot': 0.5,
            'financial_analyst_role': 0.6,
        }
    })
    """Configuration dict for the selected prompting strategy"""


@dataclass
class AgenticConfig(AdvancedConfig):
    """Configuration for System 3 Agentic RAG Pipeline."""
    
    # ===== AGENT SETTINGS =====
    AGENT_MAX_RETRIES: int = 1
    """Maximum number of retry iterations for query rewriting (total attempts = retries + 1)"""
    
    RETRY_ON_LOW_CONFIDENCE: bool = True
    """Whether to retry retrieval if grader confidence is below threshold"""
    
    GRADER_CONFIDENCE_THRESHOLD: float = 0.51
    """Minimum confidence threshold (0.0-1.0) for document relevance grading"""
    
    AGENT_DECISION_LOGGING: bool = True
    """Whether to log all agent decisions (query rewriter, grader, generator) for analysis"""
    
    AGENT_LLM_MODEL: str = "qwen3-vl:8b-instruct"  # Lightweight LLM for agent decisions (Query Rewriter, Grader, Generator strategy)