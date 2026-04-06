# src/pipelines/advanced_pipeline.py
# Advanced RAG pipeline with query technique support
# Highly modular and configurable for different embedders, LLMs, retrieval methods, etc.

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.advanced_config import AdvancedConfig
from data.data_loader import load_train_data, load_test_data
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from indexing.embedder import TextEmbedder, create_chunk_embeddings
from indexing.vector_store import QdrantConfig, QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation
from query_techniques import get_query_technique

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRAGPipeline:
    """
    Advanced RAG pipeline with configurable components and query techniques.
    
    Features:
    - Modular component management (embedder, generator, vector DB, retriever)
    - Query technique selection and configuration
    - Easy component swapping for experimentation
    - Configurable chunking, embedding, and retrieval settings
    """
    
    def __init__(self, config: 'AdvancedConfig'):
        """
        Initialize the advanced pipeline with a configuration object.
        
        Args:
            config: AdvancedConfig instance with all settings
        """
        self.config = config
        
        # Components (initialized during build_index or setup)
        self.embedder: Optional[TextEmbedder] = None
        self.vector_db: Optional[QdrantVectorDB] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.generator: Optional[BaselineGenerator] = None
        self.query_technique = None
        self.chunks: Optional[List[Dict]] = None
    
    def build_index(self, force_rebuild: bool = True):
        """
        Build or load the document index.
        
        Steps:
        1. Initialize embedder
        2. Initialize vector database
        3. Load or create chunks
        4. Create embeddings
        5. Index documents
        6. Initialize hybrid retriever
        7. Initialize query technique
        """
        logger.info("Building index with advanced configuration...")
        start_time = time.time()
        
        # Step 1: Initialize embedder with configured model
        logger.info(f"Initializing embedder: {self.config.EMBEDDING_MODEL}")
        self.embedder = TextEmbedder(self.config.EMBEDDING_MODEL)
        
        # Step 2: Initialize Qdrant with configured settings
        logger.info(f"Initializing vector database: {self.config.VECTOR_DB_MODE}")
        qdrant_config = QdrantConfig(
            db_mode=self.config.VECTOR_DB_MODE,
            local_path=self.config.VECTOR_DB_PATH,
            collection_name=self.config.VECTOR_DB_COLLECTION,
            embedding_dimension=self.config.EMBEDDING_DIMENSION,
            distance=self.config.VECTOR_DB_DISTANCE,
            top_k=self.config.TOP_K
        )
        self.vector_db = QdrantVectorDB(qdrant_config)
        
        # Check if collection exists
        collection_exists = False
        collection_document_count = 0
        
        try:
            collection_document_count = self.vector_db.count_documents()
            collection_exists = collection_document_count > 0
            logger.info(f"Collection check: {collection_document_count} documents found")
        except Exception as e:
            logger.info(f"Collection does not exist: {e}")
            collection_exists = False
        
        # Load existing index if available and not forcing rebuild
        if not force_rebuild and collection_exists:
            logger.info(f"Loading existing index with {collection_document_count} documents...")
            if self.config.USE_HYBRID_RETRIEVAL:
                chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
                if not chunks_path.is_absolute():
                    chunks_path = Path(self.config.PROJECT_ROOT) / self.config.PREPROCESSED_CHUNKS_FILE.lstrip('./').lstrip('.\\')
                self.chunks = load_preprocessed_chunks(chunks_path)
            elapsed = time.time() - start_time
            logger.info(f"Index loaded in {elapsed:.2f} seconds")
            
            # Initialize remaining components
            self._initialize_retriever()
            return
        
        # Build new index
        logger.info("Building new index...")
        
        # Step 3: Load or create chunks
        if self.config.USE_PREPROCESSED_CHUNKS:
            logger.info(f"Loading pre-processed chunks from: {self.config.PREPROCESSED_CHUNKS_FILE}")
            # Resolve path relative to project root if it's relative
            chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
            if not chunks_path.is_absolute():
                chunks_path = Path(self.config.PROJECT_ROOT) / self.config.PREPROCESSED_CHUNKS_FILE.lstrip('./').lstrip('.\\')
            self.chunks = load_preprocessed_chunks(chunks_path)
            print_chunk_statistics(self.chunks)
            logger.info(f"✓ Loaded {len(self.chunks)} chunks")
        else:
            logger.info("Processing documents from scratch...")
            # This would include PDF processing, etc.
            # For now, assuming pre-processed chunks exist
            chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
            if not chunks_path.is_absolute():
                chunks_path = Path(self.config.PROJECT_ROOT) / self.config.PREPROCESSED_CHUNKS_FILE.lstrip('./').lstrip('.\\')
            self.chunks = load_preprocessed_chunks(chunks_path)
        
        # Step 4: Create embeddings
        logger.info(f"Creating embeddings with {self.config.EMBEDDING_MODEL}...")
        embeddings = create_chunk_embeddings(self.chunks, self.embedder)
        logger.info(f"✓ Created {len(embeddings)} embeddings")
        
        # Step 5: Index in vector database
        logger.info("Indexing documents in vector database...")
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
        
        # Step 6: Initialize retriever and query technique
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize hybrid retriever and query technique."""
        if self.config.USE_HYBRID_RETRIEVAL:
            logger.info("Initializing hybrid retriever...")
            self.hybrid_retriever = HybridRetriever(
                chunks=self.chunks,
                embedder=self.embedder,
                vector_db=self.vector_db,
                top_k=self.config.TOP_K,
            )
    
    def initialize_components(self):
        """Initialize generator and query technique."""
        logger.info(f"Initializing generator: {self.config.LLM_MODEL}")
        self.generator = BaselineGenerator(self.config.OLLAMA_BASE_URL, self.config.LLM_MODEL)
        
        logger.info(f"Initializing query technique: {self.config.QUERY_TECHNIQUE}")
        self.query_technique = get_query_technique(
            self.config.QUERY_TECHNIQUE,
            self.embedder,
            self.hybrid_retriever or self.retriever,
            self.generator,
            self.config.QUERY_TECHNIQUE_CONFIG
        )
    
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the configured retrieval method.
        
        Args:
            question: User's question
            top_k: Override default top_k if needed
            
        Returns:
            List of retrieved documents
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        if self.config.USE_HYBRID_RETRIEVAL and self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(question, top_k=top_k)
        
        # Fallback to dense-only
        query_embedding = self.embedder.embed_query(question)
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k)
    
    def retrieve_with_technique(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using the selected query technique.
        
        Args:
            question: User's question
            top_k: Override default top_k if needed
            
        Returns:
            List of retrieved documents from query technique
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        if self.query_technique is None:
            raise RuntimeError("Query technique not initialized. Call initialize_components() first.")
        
        return self.query_technique.retrieve(question, top_k)
    
    def run_query(self, question: str, use_technique: bool = True, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single query through the pipeline.
        
        Args:
            question: User's question
            use_technique: Use query technique for retrieval (True) or basic retrieval (False)
            top_k: Override default top_k if needed
            
        Returns:
            Dictionary with question, retrieved context, and answer
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        # Retrieve documents
        if use_technique and self.query_technique:
            retrieved = self.retrieve_with_technique(question, top_k)
        else:
            retrieved = self.retrieve(question, top_k)
        
        # Format context
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # Generate answer
        answer = self.generator.generate(question, context)
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "num_docs": len(retrieved),
        }
    
    def evaluate(self, test_questions: List[Dict[str, Any]], use_technique: bool = True):
        """
        Evaluate pipeline on test questions (optimized with parallel answer generation).
        
        Two-phase approach:
        1. Retrieve documents for all questions (sequential, with variant generation in parallel)
        2. Generate answers for all questions in parallel (2 workers)
        
        Args:
            test_questions: List of test question dicts with 'question' and 'answer' keys
            use_technique: Use query technique for retrieval
        """
        eval_start_time = time.time()
        logger.info(f"Evaluating on {len(test_questions)} questions with query_technique={use_technique}...")
        
        test_subset = test_questions[:self.config.EVAL_SUBSET_SIZE]
        
        # ===== PHASE 1: Retrieve for all questions in parallel =====
        phase1_start = time.time()
        logger.info(f"Phase 1: Retrieving documents in parallel ({len(test_subset)} questions, {self.config.RETRIEVAL_WORKERS} workers)...")
        all_retrievals = [None] * len(test_subset)
        retrieval_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.RETRIEVAL_WORKERS) as executor:
            # Submit retrieval task for each question
            futures = {}
            for i, test_q in enumerate(test_subset):
                if use_technique and self.query_technique:
                    future = executor.submit(self.retrieve_with_technique, test_q['question'], self.config.TOP_K)
                else:
                    future = executor.submit(self.retrieve, test_q['question'], self.config.TOP_K)
                futures[future] = i
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                i = futures[future]
                retrieved = future.result()
                all_retrievals[i] = retrieved
                
                test_q = test_subset[i]
                retrieval_results.append({
                    'question': test_q['question'],
                    'retrieved': retrieved,
                    'expected_answer': test_q.get('answer', '')
                })
                
                completed += 1
                logger.info(f"  Retrieved {completed}/{len(test_subset)} questions")
        
        logger.info(f"Phase 1 complete. Retrieved for {len(all_retrievals)} questions.")
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 took {phase1_time:.2f} seconds")
        
        # ===== PHASE 2: Generate answers in parallel =====
        phase2_start = time.time()
        logger.info(f"Phase 2: Generating answers in parallel ({len(test_subset)} questions, {self.config.GENERATION_WORKERS} workers)...")
        generation_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.GENERATION_WORKERS) as executor:
            # Submit all generation tasks
            futures = []
            for i, (test_q, retrieved) in enumerate(zip(test_subset, all_retrievals)):
                context = "\n\n".join([
                    f"[Document {j+1}]:\n{result['text']}"
                    for j, result in enumerate(retrieved)
                ])
                future = executor.submit(self.generator.generate, test_q['question'], context)
                futures.append((i, test_q, future))
            
            # Collect results in order as they complete
            completed_count = 0
            for idx, (i, test_q, future) in enumerate(futures):
                answer = future.result()
                completed_count += 1
                logger.info(f"  Generated {completed_count}/{len(futures)} answers...")
                
                print(f"\n{'-'*80}")
                print(f"\nQuestion: {test_q['question']}")
                print(f"Ground Truth: {test_q['answer']}")
                print(f"Generated Answer: {answer}")
                
                generation_results.append({
                    'question': test_q['question'],
                    'answer': answer,
                    'expected_answer': test_q.get('answer', '')
                })
        
        logger.info("Phase 2 complete.")
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 took {phase2_time:.2f} seconds")
        
        # ===== Evaluate metrics =====
        retrieval_metrics = evaluate_retrieval(
            [r['retrieved'] for r in retrieval_results],
            test_subset
        )
        logger.info(f"Retrieval metrics: {retrieval_metrics}")
        
        generation_metrics = evaluate_generation(
            [r['answer'] for r in generation_results],
            [r['expected_answer'] for r in generation_results]
        )
        logger.info(f"Generation metrics: {generation_metrics}")
        
        total_time = time.time() - eval_start_time
        logger.info(f"\nTOTAL EVALUATION TIME: {total_time:.2f}s (Phase 1: {phase1_time:.2f}s + Phase 2: {phase2_time:.2f}s)")
        
        return {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'timing': {
                'total': total_time,
                'phase1': phase1_time,
                'phase2': phase2_time
            }
        }
