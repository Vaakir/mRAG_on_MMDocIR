import logging
import time
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from indexing.embedder import TextEmbedder
from indexing.base_embedder import create_chunk_embeddings
from indexing.vector_database import QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

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

    def build_index(self, force_rebuild: bool = False):
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
                    'pdf_name': str(chunk.get('pdf_name', 'unknown')),
                    'pdf_path': str(chunk.get('pdf_path', 'unknown')),
                    'chunk_id': chunk.get('chunk_id'),
                    'char_len': int(chunk.get('char_len', 0)),
                    'page_numbers': chunk.get('page_numbers'),
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
                allowed_types=getattr(self.config, "ALLOWED_CHUNK_TYPES", None),
            )

    def initialize_components(self):
        """Initialize standard generator component with config values."""
        logger.info(f"Initializing generator with base URL {self.config.OLLAMA_BASE_URL} and model {self.config.LLM_MODEL}")
        self.generator = BaselineGenerator(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.LLM_MODEL,
            api_key=self.config.OLLAMA_API_KEY,
            temperature=self.config.LLM_TEMPERATURE,
            top_p=self.config.LLM_TOP_P,
            max_tokens=self.config.LLM_MAX_TOKENS,
            max_retries=self.config.LLM_MAX_RETRIES,
            retry_delay=self.config.LLM_RETRY_DELAY
        )

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid BM25 + dense retrieval (if enabled),
        or dense-only retrieval.

        Returns:
            List of dicts with: id, score, text, payload (metadata)
        """
        top_k = top_k or self.config.TOP_K
        context_window = getattr(self.config, "CONTEXT_WINDOW", 0)
        if self.config.USE_HYBRID_RETRIEVAL and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(question, top_k=top_k, context_window=context_window)

        # Fallback: dense-only
        if self.embedder is None or self.vector_db is None:
             raise RuntimeError("Embedder or Vector DB not initialized. Call build_index() first.")

        query_embedding = self.embedder.embed_query(question)
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k,
                                       allowed_types=getattr(self.config, "ALLOWED_CHUNK_TYPES", None))

    def evaluate(self, test_data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the pipeline on test data in parallel using multithreading.
        """
        # Check subset limits from config
        subset_size = getattr(self.config, 'EVAL_SUBSET_SIZE', len(test_data))
        test_subset = test_data[:subset_size]
        
        logger.info(f"Evaluating on {len(test_subset)} test queries using multithreading (kwargs: {kwargs})...\n")
        
        eval_total_start = time.time()
        
        # Pre-allocate lists to maintain order matching test_subset
        all_retrieved = [None] * len(test_subset)
        all_predictions = [None] * len(test_subset)
        all_ground_truths = [None] * len(test_subset)
        
        # Use GENERATION_WORKERS as the primary concurrency bottleneck limiter
        workers = getattr(self.config, 'GENERATION_WORKERS', 2)
        
        # ===== PHASE 1: PARALLEL RETRIEVAL & GENERATION =====
        phase1_start = time.time()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all questions to the executor using run_query
            futures = {
                executor.submit(self.run_query, record["question"], record=record, **kwargs): (i, record)
                for i, record in enumerate(test_subset)
            }
            
            completed = 0
            for future in as_completed(futures):
                i, record = futures[future]
                result = future.result()
                
                # Store results at correct original index
                all_retrieved[i] = result['retrieved_docs']
                all_predictions[i] = result['answer']
                all_ground_truths[i] = record['answer']
                
                completed += 1
                logger.info(f"Processed query {completed}/{len(test_subset)}")
                logger.info(f"Question: {record['question']}")
                logger.info(f"Ground Truth: {record['answer']}")
                logger.info(f"Generated Answer: {result['answer']}\n")
        
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 (Parallel Retrieval + Generation): {phase1_time:.2f}s")
        
        # ===== PHASE 2: METRIC COMPUTATION =====
        phase2_start = time.time()
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_subset)
        generation_metrics = evaluate_generation(all_predictions, all_ground_truths)
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        
        eval_total_time = time.time() - eval_total_start
        
        all_metrics = {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'timing': {
                'phase1': phase1_time,
                'phase2': phase2_time,
                'total': eval_total_time
            }
        }
        
        return all_metrics
