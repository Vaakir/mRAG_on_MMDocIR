# src/pipelines/baseline_pipeline.py
# Updated to use team's integrated components:
# - unstructured for PDF processing
# - Team's Chunking class with fixed_size strategy
# - Qdrant vector database
# - CLIP embeddings (unchanged)

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from config.config import *
from data.data_loader import load_train_data, load_test_data
from data.pdf_processor import process_all_pdfs
# from data.pdf_cache import PDFCache
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from preprocessing.chunker import chunk_documents
# from preprocessing.text_cleaner import filter_chunks
from indexing.embedder import TextEmbedder, create_chunk_embeddings
from indexing.vector_store import QdrantConfig, QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineRAGPipeline:
    """
    Complete baseline RAG pipeline using team's integrated components.
    
    Components:
    - PDF Processing: unstructured library (structured blocks)
    - Chunking: Team's fixed_size strategy (1000 chars)
    - Embeddings: CLIP ViT-B/32 (512-dim)
    - Vector DB: Qdrant (local mode)
    - Generation: Ollama API
    """
    
    def __init__(self):
        self.embedder = None
        self.vector_db = None
        self.hybrid_retriever = None
        self.generator = None
        self.chunks = None
    
    def build_index(self, force_rebuild: bool = True):
        """
        Build or load the document index using Qdrant.
        
        Steps:
        1. Extract text from PDFs (unstructured)
        2. Chunk documents (team's fixed_size strategy)
        3. Create embeddings (CLIP)
        4. Index in Qdrant
        """
        logger.info("Building index...")
        start_time = time.time()
        
        # Initialize embedder
        self.embedder = TextEmbedder(EMBEDDING_MODEL)
        
        # Initialize Qdrant
        qdrant_config = QdrantConfig(
            db_mode=VECTOR_DB_MODE,
            local_path=VECTOR_DB_PATH,
            collection_name=VECTOR_DB_COLLECTION,
            embedding_dimension=EMBEDDING_DIMENSION,
            distance=VECTOR_DB_DISTANCE,
            top_k=TOP_K
        )
        self.vector_db = QdrantVectorDB(qdrant_config)
        
        # Check if collection exists
        if not force_rebuild:
            try:
                count = self.vector_db.count_documents()
                if count > 0:
                    logger.info(f"Loading existing index with {count} documents...")
                    # Still need chunks in memory for BM25
                    if USE_HYBRID_RETRIEVAL:
                        self.chunks = load_preprocessed_chunks(PREPROCESSED_CHUNKS_FILE)
                    elapsed = time.time() - start_time
                    logger.info(f"Index loaded in {elapsed:.2f} seconds")
            except Exception as e:
                logger.info(f"No existing collection found ({e}), building new index...")
                force_rebuild = True
        
        if force_rebuild:
            logger.info("Building new index...")
            
            # Check if we should use pre-processed chunks
            if USE_PREPROCESSED_CHUNKS:
                logger.info("=" * 80)
                logger.info("USING PRE-PROCESSED CHUNKS FROM TEAM")
                logger.info("=" * 80)
                logger.info(f"Loading chunks from: {PREPROCESSED_CHUNKS_FILE}")
                
                # Load pre-processed chunks directly (skip PDF extraction and chunking)
                self.chunks = load_preprocessed_chunks(PREPROCESSED_CHUNKS_FILE)
                print_chunk_statistics(self.chunks)
                
                logger.info(f"✓ Loaded {len(self.chunks)} pre-processed chunks")
            else:
                # Original flow: Extract PDFs → Chunk → Filter
                logger.info("Processing PDFs from scratch (USE_PREPROCESSED_CHUNKS=False)")
                
                # Initialize PDF cache
                pdf_cache = PDFCache(CACHE_DIR)
                
                # Step 1: Extract text from PDFs (or load from cache)
                logger.info("Step 1: Extracting text from PDFs...")
                
                # Check if we have cached PDFs
                if pdf_cache.exists() and not force_rebuild:
                    logger.info("Found cached processed PDFs, loading...")
                    documents = pdf_cache.load()
                    if documents:
                        logger.info(f"✓ Loaded {len(documents)} PDFs from cache")
                    else:
                        logger.info("Cache load failed, processing PDFs...")
                        documents = process_all_pdfs(PDF_DIR)
                        pdf_cache.save(documents)
                else:
                    logger.info("Processing PDFs from source...")
                    documents = process_all_pdfs(PDF_DIR)
                    logger.info(f"Extracted {len(documents)} PDFs")
                    
                    # Save to cache for future inspection
                    logger.info("Saving processed PDFs to cache...")
                    pdf_cache.save(documents, metadata={
                        'source_dir': str(PDF_DIR),
                        'processing_method': 'pdfplumber with unstructured fallback'
                    })
                
                logger.info(f"✓ Total documents: {len(documents)}")

                
                # Step 2: Chunk documents using team's strategy
                logger.info("Step 2: Chunking documents (fixed_size strategy)...")
                self.chunks = chunk_documents(
                    documents,
                    strategy=CHUNKING_STRATEGY,
                    max_chars=CHUNK_SIZE
                )
                logger.info(f"Created {len(self.chunks)} chunks")
                
                # Step 2.5: Filter out low-quality/noisy chunks
                logger.info("Step 2.5: Filtering noisy chunks...")
                original_count = len(self.chunks)
                self.chunks = filter_chunks(self.chunks)
                filtered_count = original_count - len(self.chunks)
                logger.info(f"Filtered out {filtered_count} noisy chunks ({filtered_count/original_count*100:.1f}%)")
                logger.info(f"Retained {len(self.chunks)} high-quality chunks")
            
            # Step 3: Create embeddings using CLIP
            logger.info("Step 3: Creating CLIP embeddings...")
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
                        'char_len': chunk['char_len']
                    }
                })
            
            self.vector_db.index_documents(qdrant_docs)

            elapsed = time.time() - start_time
            logger.info(f"Index built in {elapsed:.2f} seconds")

        # Build hybrid retriever (BM25 + dense) — always, using loaded chunks
        if USE_HYBRID_RETRIEVAL:
            logger.info("Building hybrid BM25 + dense retriever...")
            self.hybrid_retriever = HybridRetriever(
                chunks=self.chunks,
                embedder=self.embedder,
                vector_db=self.vector_db,
                top_k=TOP_K,
            )
    
    def initialize_components(self):
        """Initialize generator."""
        self.generator = BaselineGenerator(OLLAMA_BASE_URL, LLM_MODEL)
    
    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid BM25 + dense retrieval (if enabled),
        or dense-only retrieval.

        Returns:
            List of dicts with: id, score, text, payload (metadata)
        """
        if USE_HYBRID_RETRIEVAL and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(question, top_k=top_k)

        # Fallback: dense-only
        query_embedding = self.embedder.embed_query(question)
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k)
    
    def run_query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Run a single query through the pipeline."""
        # Retrieve
        retrieved = self.retrieve(question, top_k)
        
        # Format context
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # Generate
        answer = self.generator.generate(question, context)
        
        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer
        }
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data.
        
        Returns metrics:
        - Retrieval: precision@k, recall@k
        - Generation: exact_match, token_f1
        """
        logger.info(f"Evaluating on {len(test_data)} test queries...")
        
        all_retrieved = []
        all_predictions = []
        all_ground_truths = []
        
        for i, record in enumerate(test_data, 1):
            logger.info(f"Processing query {i}/{len(test_data)}")
            
            # Run query
            result = self.run_query(record["question"])
            print(f"Question: {record['question']}")
            print(f"Ground Truth: {record['answer']}")
            print(f"Generated Answer: {result['answer']}")

            # Collect retrieved docs (keep as list of dicts for eval)
            all_retrieved.append(result['retrieved_docs'])
            
            # Collect predictions and ground truths
            all_predictions.append(result['answer'])
            all_ground_truths.append(record['answer'])
        
        # Evaluate retrieval
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_data)
        
        # Evaluate generation
        generation_metrics = evaluate_generation(
            all_predictions, 
            all_ground_truths
        )
        
        # Combine metrics
        all_metrics = {**retrieval_metrics, **generation_metrics}
        
        # Log results
        logger.info("=== EVALUATION RESULTS ===")
        for metric, value in all_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return all_metrics
