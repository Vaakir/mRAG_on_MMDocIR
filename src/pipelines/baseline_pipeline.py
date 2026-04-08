# src/pipelines/baseline_pipeline.py
# Updated to use team's integrated components:
# - unstructured for PDF processing
# - Team's Chunking class with fixed_size strategy
# - Qdrant vector database
# - Jina CLIP v2 for embeddings (1024D, multimodal text + image shared space)
# - Ollama API for generation (qwen3:32b for strong reasoning)

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from config.config import BaselineConfig
from data.data_loader import load_train_data, load_test_data
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
# from preprocessing.text_cleaner import filter_chunks
from indexing.embedder import TextEmbedder, create_chunk_embeddings
from indexing.vector_database import QdrantConfig, QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
class BaselineRAGPipeline:
    """
    Complete baseline RAG pipeline using team's integrated components.
    
    Components:
    - PDF Processing: unstructured library (structured blocks)
    - Chunking: Team's fixed_size strategy (1000 chars)
    - Embeddings: Jina CLIP v2 (1024D, text-only for System 1, ready for multimodal in System 2)
    - Vector DB: Qdrant (local mode)
    - Generation: Ollama API
    """
    #-------------------
    def __init__(self, config=None):
        self.config = config or BaselineConfig()
        self.embedder = None          # TextEmbedder instance for creating embeddings
        self.vector_db = None         # QdrantVectorDB instance for indexing and retrieval
        self.hybrid_retriever = None  # HybridRetriever instance for combining multiple retrieval methods
        self.generator = None         # BaselineGenerator instance for generating responses
        self.chunks = None            # List of pre-processed document chunks
    #-------------------
    def build_index(self, force_rebuild: bool = True):
        """
        Build or load the document index using Qdrant.
        
        Steps:
        1. Extract text from PDFs (unstructured)
        2. Chunk documents (team's fixed_size strategy)
        3. Create embeddings
        4. Index in Qdrant
        """
        logger.info("Building index...")
        start_time = time.time() # Start timer to measure index-building time
        
        # Initialize the embedder
        self.embedder = TextEmbedder(self.config.EMBEDDING_MODEL)
        
        # Initialize Qdrant
        qdrant_config = QdrantConfig(
            db_mode=self.config.VECTOR_DB_MODE,
            local_path=self.config.VECTOR_DB_PATH,
            collection_name=self.config.VECTOR_DB_COLLECTION,
            embedding_dimension=self.config.EMBEDDING_DIMENSION,
            distance=self.config.VECTOR_DB_DISTANCE,
            top_k=self.config.TOP_K
        )
        self.vector_db = QdrantVectorDB(qdrant_config)
        
        # Check if collection exists and has documents
        collection_exists = False
        collection_document_count = 0
        
        try:
            collection_document_count = self.vector_db.count_documents() # Check if collection exists and has documents
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
                self.chunks = load_preprocessed_chunks(self.config.PREPROCESSED_CHUNKS_FILE)
                
            elapsed = time.time() - start_time # Measure time taken to load the index
            logger.info(f"Index loaded in {elapsed:.2f} seconds")
            return  # EXIT early - don't rebuild if valid collection exists
        
        # If we reach here: either force_rebuild=True OR collection is empty/missing
        # Either way, we need to build the index
        force_rebuild = True
        
        if force_rebuild:
            logger.info("Building new index...")
            
            logger.info("=" * 80)
            logger.info("USING PRE-PROCESSED CHUNKS")
            logger.info("=" * 80)
            logger.info(f"Loading chunks from: {self.config.PREPROCESSED_CHUNKS_FILE}")
            
            # Load pre-processed chunks directly
            self.chunks = load_preprocessed_chunks(self.config.PREPROCESSED_CHUNKS_FILE)
            print_chunk_statistics(self.chunks)
            
            logger.info(f"[OK] Loaded {len(self.chunks)} pre-processed chunks")
            
            # Step 3: Create embeddings using Jina CLIP v2 (1024D, text-only for System 1)
            logger.info("Step 3: Creating Jina CLIP v2 embeddings (text-only)...")
            embeddings = create_chunk_embeddings(self.chunks, self.embedder)
            logger.info(f"Created {len(embeddings)} embeddings")
            
            # Step 4: Create Qdrant collection and index documents
            logger.info("Step 4: Indexing in Qdrant...")
            self.vector_db.create_collection(force_recreate=True) 
            
            # Prepare documents for Qdrant
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

        # Build hybrid retriever (BM25 + dense) — always, using loaded chunks
        if self.config.USE_HYBRID_RETRIEVAL:
            logger.info("Building hybrid BM25 + dense retriever...")
            self.hybrid_retriever = HybridRetriever(
                chunks=self.chunks,
                embedder=self.embedder,
                vector_db=self.vector_db,
                top_k=self.config.TOP_K,
            )
    #-------------------
    def initialize_components(self):
        """Initialize generator."""
        self.generator = BaselineGenerator(self.config.OLLAMA_BASE_URL, self.config.LLM_MODEL)
    #-------------------
    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
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
        query_embedding = self.embedder.embed_query(question)
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k)
    #-------------------
    def run_query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Run a single query through the pipeline with timing instrumentation.
        
        Tracks:
        - Retrieval time
        - Generation time
        - Total query time
        
        Returns:
            Dict with question, retrieved_docs, context, answer, and timing info
        """
        top_k = top_k or self.config.TOP_K
        query_total_start = time.time()

        
        # ===== STEP 1: RETRIEVAL =====
        retrieval_start = time.time()
        retrieved = self.retrieve(question, top_k)
        retrieval_time = time.time() - retrieval_start
        
        # ===== STEP 2: CONTEXT FORMATTING =====
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # ===== STEP 3: GENERATION =====
        generation_start = time.time()
        answer = self.generator.generate(question, context)
        generation_time = time.time() - generation_start
        
        query_total_time = time.time() - query_total_start
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": query_total_time
            }
        }
    #-------------------
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data with comprehensive timing instrumentation.
        """
        logger.info(f"Evaluating on {len(test_data)} test queries...\n")
        
        eval_total_start = time.time()
        
        all_retrieved = []
        all_predictions = []
        all_ground_truths = []
        
        # ===== PHASE 1: RETRIEVAL & GENERATION =====
        phase1_start = time.time()
        
        for i, record in enumerate(test_data, 1):
            logger.info(f"Processing query {i}/{len(test_data)}")
            result = self.run_query(record["question"])

            logger.info(f"Question: {record['question']}")
            logger.info(f"Ground Truth: {record['answer']}")
            logger.info(f"Generated Answer: {result['answer']}\n")

            all_retrieved.append(result['retrieved_docs'])
            all_predictions.append(result['answer'])
            all_ground_truths.append(record['answer'])
        
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 (Retrieval + Generation): {phase1_time:.2f}s")
        
        # ===== PHASE 2: METRIC COMPUTATION =====
        phase2_start = time.time()
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_data)
        generation_metrics = evaluate_generation(all_predictions, all_ground_truths)
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        
        eval_total_time = time.time() - eval_total_start
        
        all_metrics = {
            **retrieval_metrics,
            **generation_metrics,
            'timing': {
                'phase1': phase1_time,
                'phase2': phase2_time,
                'total': eval_total_time
            }
        }
        
        print("\n === EVALUATION RESULTS ===")
        for metric_key, metric_value in all_metrics.items():
            if metric_key != 'timing':
                if isinstance(metric_value, (int, float)):
                    print(f"{metric_key}: {metric_value:.4f}")
                else:
                    print(f"{metric_key}: {metric_value}")
                    
        print("\n === TIMING BREAKDOWN ===")
        print(f"Phase 1 (Query Processing): {phase1_time:.2f}s")
        print(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        print(f"Total Evaluation Time: {eval_total_time:.2f}s")
        
        return all_metrics