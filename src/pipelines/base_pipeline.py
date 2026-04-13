import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from indexing.embedder import TextEmbedder, create_chunk_embeddings
from indexing.vector_database import QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRAGPipeline:
    """
    Base RAG pipeline standardizing component initialization, indexing, and retrieval.
    Inherited by both BaselineRAGPipeline and AdvancedRAGPipeline.
    """
    
    def __init__(self, config):
        """
        Initialize the base pipeline with a configuration object.
        """
        self.config = config
        
        # Core Components
        self.embedder: Optional[TextEmbedder] = None
        self.vector_db: Optional[QdrantVectorDB] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.generator: Optional[BaselineGenerator] = None
        self.chunks: Optional[List[Dict]] = None

    def build_index(self, force_rebuild: bool = True):
        """
        Build or load the document index using Qdrant.
        
        Steps:
        1. Initialize embedder
        2. Initialize vector database
        3. Load or create chunks
        4. Create embeddings
        5. Index documents
        6. Initialize hybrid retriever
        """
        logger.info(f"Building index with configuration: {type(self.config).__name__}...")
        start_time = time.time()
        
        # Step 1: Initialize embedder
        logger.info(f"Initializing embedder: {self.config.EMBEDDING_MODEL}")
        self.embedder = TextEmbedder(self.config.EMBEDDING_MODEL)
        
        # Step 2: Initialize Qdrant
        logger.info(f"Initializing vector database: {self.config.VECTOR_DB_MODE}")
        self.vector_db = QdrantVectorDB(self.config)
        
        # Check if collection exists and has documents
        collection_exists = False
        collection_document_count = 0
        
        try:
            collection_document_count = self.vector_db.count_documents()
            collection_exists = collection_document_count > 0
            logger.info(f"Collection check: {collection_document_count} documents found in Qdrant")
        except Exception as e:
            logger.info(f"Collection does not exist or cannot be accessed: {e}")
            collection_exists = False
            collection_document_count = 0
        
        # If not forcing rebuild AND collection has documents, load it
        if not force_rebuild and collection_exists:
            logger.info(f"Loading existing index with {collection_document_count} documents...")
            
            # Still need chunks in memory for BM25
            if self.config.USE_HYBRID_RETRIEVAL:
                chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
                if not chunks_path.is_absolute():
                    chunks_path = Path(getattr(self.config, 'PROJECT_ROOT', '.')) / self.config.PREPROCESSED_CHUNKS_FILE.lstrip('./').lstrip('.\\')
                
                try:
                    self.chunks = load_preprocessed_chunks(chunks_path)
                except FileNotFoundError:
                    # Fallback path if PROJECT_ROOT wasn't needed or misconfigured
                    self.chunks = load_preprocessed_chunks(self.config.PREPROCESSED_CHUNKS_FILE)
                
            elapsed = time.time() - start_time
            logger.info(f"Index loaded in {elapsed:.2f} seconds")
            
            # Initialize remaining components
            self._initialize_retriever()
            return  # EXIT early
        
        logger.info("Building new index...")
        
        logger.info("=" * 80)
        logger.info("USING PRE-PROCESSED CHUNKS")
        logger.info("=" * 80)
        
        # Step 3: Load chunks
        chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
        if not chunks_path.is_absolute():
            chunks_path = Path(getattr(self.config, 'PROJECT_ROOT', '.')) / self.config.PREPROCESSED_CHUNKS_FILE.lstrip('./').lstrip('.\\')
            
        logger.info(f"Loading chunks from: {chunks_path}")
        
        try:
            self.chunks = load_preprocessed_chunks(chunks_path)
        except FileNotFoundError:
            self.chunks = load_preprocessed_chunks(self.config.PREPROCESSED_CHUNKS_FILE)
            
        print_chunk_statistics(self.chunks)
        logger.info(f"[OK] Loaded {len(self.chunks)} pre-processed chunks")
        
        # Step 4: Create embeddings
        logger.info(f"Step 4: Creating embeddings using {self.config.EMBEDDING_MODEL}...")
        embeddings = create_chunk_embeddings(self.chunks, self.embedder)
        logger.info(f"Created {len(embeddings)} embeddings")
        
        # Step 5: Create Qdrant collection and index documents
        logger.info("Step 5: Indexing in Qdrant...")
        self.vector_db.create_collection(force_recreate=True) 
        
        qdrant_docs = []
        for i, (chunk, embedding) in enumerate(zip(self.chunks, embeddings)): 
            qdrant_docs.append({
                'id': i,
                'embedding': embedding,
                'text': chunk['text'],
                'metadata': {
                    'pdf_name': chunk['pdf_name'],
                    'pdf_path': chunk['pdf_path'],
                    'chunk_id': chunk['chunk_id'],
                    'char_len': chunk['char_len'],
                    'page_numbers': chunk['page_numbers'],
                }
            })
        
        self.vector_db.index_documents(qdrant_docs)

        elapsed = time.time() - start_time
        logger.info(f"Index built in {elapsed:.2f} seconds")

        # Step 6: Initialize hybrid retriever
        self._initialize_retriever()
        
    def _initialize_retriever(self):
        """Initialize hybrid retriever."""
        if self.config.USE_HYBRID_RETRIEVAL:
            logger.info("Building hybrid BM25 + dense retriever...")
            self.hybrid_retriever = HybridRetriever(
                chunks=self.chunks,
                embedder=self.embedder,
                vector_db=self.vector_db,
                top_k=self.config.TOP_K,
            )

    def initialize_components(self):
        """Initialize standard generator component."""
        logger.info(f"Initializing generator with base URL {self.config.OLLAMA_BASE_URL} and model {self.config.LLM_MODEL}")
        self.generator = BaselineGenerator(self.config.OLLAMA_BASE_URL, self.config.LLM_MODEL)

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid BM25 + dense retrieval (if enabled),
        or dense-only retrieval.

        Returns:
            List of dicts with: id, score, text, payload (metadata)
        """
        top_k = top_k or self.config.TOP_K
        if self.config.USE_HYBRID_RETRIEVAL and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(question, top_k=top_k)

        # Fallback: dense-only
        if self.embedder is None or self.vector_db is None:
             raise RuntimeError("Embedder or Vector DB not initialized. Call build_index() first.")

        query_embedding = self.embedder.embed_query(question)
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k)
