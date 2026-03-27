# src/pipelines/baseline_pipeline.py

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from config.config import *
from data.data_loader import load_train_data, load_test_data
from data.pdf_processor import process_all_pdfs
from preprocessing.chunker import chunk_documents
from indexing.embedder import TextEmbedder, create_chunk_embeddings
from indexing.vector_store import FAISSVectorStore
from retrieval.retriever import BaselineRetriever
from generation.generator import BaselineGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineRAGPipeline:
    """Complete baseline RAG pipeline."""
    
    def __init__(self):
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.chunks = None
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load the document index."""
        
        # Check if index already exists
        if not force_rebuild and FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
            logger.info("Loading existing index...")
            self.embedder = TextEmbedder(EMBEDDING_MODEL)
            self.vector_store = FAISSVectorStore(EMBEDDING_DIMENSION)
            self.vector_store.load(FAISS_INDEX_PATH, CHUNKS_PATH)
            self.chunks = self.vector_store.chunks
            return
        
        logger.info("Building new index...")
        start_time = time.time()
        
        # Step 1: Extract text from PDFs
        logger.info("Step 1: Extracting text from PDFs...")
        documents = process_all_pdfs(PDF_DIR)
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        self.chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Step 3: Create embeddings
        logger.info("Step 3: Creating embeddings...")
        self.embedder = TextEmbedder(EMBEDDING_MODEL)
        embeddings = create_chunk_embeddings(self.chunks, self.embedder)
        
        # Step 4: Build FAISS index
        logger.info("Step 4: Building FAISS index...")
        self.vector_store = FAISSVectorStore(EMBEDDING_DIMENSION)
        self.vector_store.add_embeddings(embeddings, self.chunks)
        
        # Save index
        self.vector_store.save(FAISS_INDEX_PATH, CHUNKS_PATH)
        
        elapsed = time.time() - start_time
        logger.info(f"Index built in {elapsed:.2f} seconds")
    
    def initialize_components(self):
        """Initialize retriever and generator."""
        self.retriever = BaselineRetriever(self.embedder, self.vector_store)
        self.generator = BaselineGenerator(OLLAMA_BASE_URL, LLM_MODEL)
    
    def run_query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """Run a single query through the pipeline."""
        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k)
        context = self.retriever.retrieve_context(question, top_k)
        
        # Generate
        answer = self.generator.generate(question, context)
        
        return {
            "question": question,
            "retrieved_chunks": [r[0] for r in retrieved],
            "retrieved_scores": [r[1] for r in retrieved],
            "context": context,
            "answer": answer
        }
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the pipeline on test data."""
        logger.info(f"Evaluating on {len(test_data)} test queries...")
        
        all_retrieved = []
        all_predictions = []
        all_ground_truths = []
        
        for i, record in enumerate(test_data):
            if i % 10 == 0:
                logger.info(f"Processing query {i+1}/{len(test_data)}")
            
            result = self.run_query(record["question"])
            
            all_retrieved.append(result["retrieved_chunks"])
            all_predictions.append(result["answer"])
            all_ground_truths.append(record["answer"])
        
        # Compute metrics
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_data)
        generation_metrics = evaluate_generation(all_predictions, all_ground_truths)
        
        all_metrics = {**retrieval_metrics, **generation_metrics}
        
        logger.info("=== EVALUATION RESULTS ===")
        for metric, value in all_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return all_metrics