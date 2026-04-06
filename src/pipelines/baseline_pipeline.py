import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from config.config import *
from data.data_loader import load_train_data, load_test_data
from data.pdf_processor import process_all_pdfs
# from data.pdf_cache import PDFCache
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
# from generated_stuff.pdf_cache import PDFCache
from preprocessing.text_cleaner import filter_chunks
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

# -------------------------------------------------------------------
class BaselineRAGPipeline:
    """
    Complete baseline RAG pipeline.
    
    Components:
    - PDF Processing: unstructured library (structured blocks)
    - Chunking: Fixed-size strategy (1000 chars)
    - Embeddings: Jina CLIP v2 (1024D, text-only for System 1, ready for multimodal in System 2)
    - Vector DB: Qdrant (local mode)
    - Generation: Ollama API
    """
    #-------------------
    def __init__(self):
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
        2. Chunk documents (fixed_size strategy)
        3. Create embeddings
        4. Index in Qdrant
        """
        logger.info("Building index...")
        start_time = time.time() # Start timer to measure index-building time
        
        # Initialize the embedder
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
            if USE_HYBRID_RETRIEVAL:
                self.chunks = load_preprocessed_chunks(PREPROCESSED_CHUNKS_FILE)
                
            elapsed = time.time() - start_time # Measure time taken to load the index
            logger.info(f"Index loaded in {elapsed:.2f} seconds")
            return  # EXIT early (don't rebuild if valid collection exists)
        
        # If we reach here: either force_rebuild=True OR collection is empty/missing
        # Either way, we need to build the index
        force_rebuild = True
        
        if force_rebuild:
            logger.info("Building new index...")
            
            # Check if we should use pre-processed chunks
            if USE_PREPROCESSED_CHUNKS: # If set to True, then we just load the pre-processed chunks directly 
                logger.info("=" * 80)
                logger.info("USING PRE-PROCESSED CHUNKS")
                logger.info("=" * 80)
                logger.info(f"Loading chunks from: {PREPROCESSED_CHUNKS_FILE}")
                
                # Load pre-processed chunks directly (i.e. skip PDF extraction and chunking)
                self.chunks = load_preprocessed_chunks(PREPROCESSED_CHUNKS_FILE)
                print_chunk_statistics(self.chunks)
                
                logger.info(f"[OK] Loaded {len(self.chunks)} pre-processed chunks")
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
                    documents = pdf_cache.load() # Load the processed PDF documents from cache if it exists and we're not forcing a rebuild (this allows us to skip the potentially time-consuming PDF processing step if we've already done it once and saved the results)
                    if documents: # If we successfully loaded documents from cache, we can proceed with those instead of re-processing the PDFs (this is a big time saver for development and testing)
                        logger.info(f"[OK] Loaded {len(documents)} PDFs from cache")
                    else: # If cache load failed (e.g. corrupted cache), we fall back to processing the PDFs from source and then save to cache for future runs (this ensures we can recover gracefully if the cache is not usable for some reason, while still benefiting from caching in subsequent runs)
                        logger.info("Cache load failed, processing PDFs...")
                        documents = process_all_pdfs(PDF_DIR) # Process the PDFs from the source directory using the defined PDF processing function (this is where we extract text and metadata from the raw PDF files, which can be time-consuming, hence the importance of caching)
                        pdf_cache.save(documents) # Save the processed documents to cache for future runs (this allows us to avoid re-processing the PDFs every time, which can save a lot of time during development and testing when we may be iterating on other parts of the pipeline)
                else: # If no cache exists or we're forcing a rebuild, we process the PDFs from source and then save to cache for future runs (this ensures we have a fresh processing of the PDFs when needed, while still benefiting from caching in subsequent runs)
                    logger.info("Processing PDFs from source...")
                    documents = process_all_pdfs(PDF_DIR) # Process the PDFs from the source directory using the defined PDF processing function (this is where we extract text and metadata from the raw PDF files, which can be time-consuming, hence the importance of caching)
                    logger.info(f"[OK] Extracted {len(documents)} PDFs")
                    
                    # Save to cache for future inspection
                    logger.info("Saving processed PDFs to cache...")
                    pdf_cache.save(documents, metadata={
                        'source_dir': str(PDF_DIR),
                        'processing_method': 'pdfplumber with unstructured fallback'
                    })
                
                logger.info(f"[OK] Total documents: {len(documents)}")

                
                # Step 2: Chunk documents using the chosen chunking strategy (CHUNKING_STRATEGY) and chunk size (CHUNK_SIZE))
                logger.info("Step 2: Chunking documents (fixed_size strategy)...")
                self.chunks = chunk_documents( # Chunking function to create chunks from the extracted PDF documents using the specified chunking strategy (fixed_size) and chunk size (1000 characters). This will break down the long PDF texts into smaller, manageable chunks that can be embedded and indexed effectively. The resulting chunks will also include metadata such as source PDF name, path, chunk ID, character length, and page numbers for reference during retrieval and generation.
                    documents, # The list of processed PDF documents to be chunked, where each document contains the extracted text and associated metadata (e.g. PDF name, path, page numbers)
                    strategy=CHUNKING_STRATEGY, # The chunking strategy to use (in this case, 'fixed_size' which creates chunks of a fixed number of characters)
                    max_chars=CHUNK_SIZE # The maximum number of characters per chunk (1000 characters as per the implementation), which determines how the text from the PDFs will be split into chunks for embedding and indexing
                )
                logger.info(f"Created {len(self.chunks)} chunks")
                
                # Step 2.5: Filter out low-quality/noisy chunks (NOTE/OBS: not sure if this is necessary)
                logger.info("Step 2.5: Filtering noisy chunks...")
                original_count = len(self.chunks)                  # Store the original count of chunks before filtering for reference in logging
                self.chunks = filter_chunks(self.chunks)           # Use the cleaning function to filter out low-quality or noisy chunks based on heuristics such as character length, presence of meaningful text, etc. This step helps improve the quality of the chunks that will be embedded and indexed, which can lead to better retrieval and generation performance. The filtering criteria can be adjusted in the filter_chunks function as needed to find the right balance between retaining useful information and removing noise.
                filtered_count = original_count - len(self.chunks) # Calculate the number of chunks that were filtered out
                logger.info(f"Filtered out {filtered_count} noisy chunks ({filtered_count/original_count*100:.1f}%)")
                logger.info(f"Retained {len(self.chunks)} high-quality chunks")
            
            # Step 3: Create embeddings using Jina CLIP v2 (1024D, text-only for System 1)
            logger.info("Step 3: Creating Jina CLIP v2 embeddings (text-only)...")
            embeddings = create_chunk_embeddings(self.chunks, self.embedder)
            logger.info(f"Created {len(embeddings)} embeddings")
            
            # Step 4: Create Qdrant collection and index documents
            logger.info("Step 4: Indexing in Qdrant...")
            self.vector_db.create_collection(force_recreate=True) # Create the Qdrant collection for indexing, with force_recreate=True to ensure we start with a fresh collection (this will delete any existing collection with the same name and create a new one, which is useful during development and testing to avoid issues with stale or corrupted data in the collection)
            
            # Prepare documents for Qdrant
            qdrant_docs = []
            for i, (chunk, embedding) in enumerate(zip(self.chunks, embeddings)): # Iterate over the chunks and their corresponding embeddings to prepare the documents for indexing in Qdrant. Each document will include an ID, the embedding vector, the chunk text, and metadata such as PDF name, path, chunk ID, character length, and page numbers. This structured format allows us to efficiently index and retrieve the chunks based on their embeddings and associated metadata during query time.
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
            
            self.vector_db.index_documents(qdrant_docs) # Index the prepared documents in Qdrant

            elapsed = time.time() - start_time # Measure the total time taken to build the index (including PDF processing, chunking, embedding, and indexing) for reference and monitoring purposes
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
    #-------------------
    def initialize_components(self):
        """Initialize generator."""
        self.generator = BaselineGenerator(OLLAMA_BASE_URL, LLM_MODEL) # Initialize the generator component using the specified Ollama base URL and LLM model (this will allow us to generate responses based on the retrieved context during query time)
    #-------------------
    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid BM25 + dense retrieval (if enabled),
        or dense-only retrieval.

        Returns:
            List of dicts with: id, score, text, payload (metadata)
        """
        if USE_HYBRID_RETRIEVAL and self.hybrid_retriever is not None: # If hybrid retrieval is enabled and the hybrid retriever is initialized, use it to retrieve relevant chunks based on the question. The hybrid retriever will combine BM25 retrieval based on the chunk texts and dense retrieval based on the embeddings to return a ranked list of relevant chunks for the given question. This allows us to leverage both lexical and semantic matching for improved retrieval performance.
            return self.hybrid_retriever.retrieve(question, top_k=top_k) # 

        # Fallback: dense-only
        query_embedding = self.embedder.embed_query(question) # Embed the question using the embedder to get the query embedding vector, which will be used for dense retrieval in Qdrant. This allows us to retrieve chunks that are semantically similar to the question based on their embeddings, even if they don't have strong lexical overlap.
        return self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k) # Retrieve relevant chunks from Qdrant based on the query embedding, returning the top_k most similar chunks according to the specified distance metric (e.g. cosine similarity). The retrieved results will include the chunk text and associated metadata for use in the generation step.
    #-------------------
    def run_query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Run a single query through the pipeline with timing instrumentation.
        
        Tracks:
        - Retrieval time
        - Generation time
        - Total query time
        
        Returns:
            Dict with question, retrieved_docs, context, answer, and timing info
        """
        query_total_start = time.time() # Track time for total time taken for processing this query through the entire pipeline (including retrieval and generation) for reference and monitoring purposes
        
        # ===== STEP 1: RETRIEVAL =====
        retrieval_start = time.time() # Start time for measurement of retrieval time for this query
        retrieved = self.retrieve(question, top_k)
        retrieval_time = time.time() - retrieval_start # Time taken for retrieval for this query (this includes the time to embed the query and retrieve relevant chunks from Qdrant, as well as any additional processing in the hybrid retriever if enabled)
        
        # ===== STEP 2: CONTEXT FORMATTING =====
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(retrieved)
        ])
        
        # ===== STEP 3: GENERATION =====
        generation_start = time.time() # Track time for measurement of generation time for this query 
        answer = self.generator.generate(question, context)
        generation_time = time.time() - generation_start # Time taken for generation for this query (this includes the time to send the request to the Ollama API and receive the generated response based on the question and retrieved context)
        
        query_total_time = time.time() - query_total_start # Time taken for total query processing time for this query (including both retrieval and generation)
        
        # Return the results along with timing information for retrieval, generation, and total query processing time for reference and potential use in logging or analysis during evaluation
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
        
        Returns metrics:
        - Retrieval: precision@k, recall@k, MAP, MRR, NDCG@k, page_recall@k
        - Generation: exact_match, token_f1, bleu, semantic_similarity
        - Timing: phase1 (retrieval), phase2 (generation), total
        """
        logger.info(f"Evaluating on {len(test_data)} test queries... \n")
        
        eval_total_start = time.time()  # Track OVERALL evaluation time
        
        all_retrieved = []     # List to hold retrieved documents for all queries for retrieval evaluation
        all_predictions = []   # List to hold generated answers for all queries for generation evaluation
        all_ground_truths = [] # List to hold ground truth answers for all queries for generation evaluation
        
        # ===== PHASE 1: RETRIEVAL & GENERATION (per-query) =====
        phase1_start = time.time()  # Track time for retrieval phase
        
        # Loop through each test query (i.e. query from test_data) and run it through the pipeline to collect retrieved documents, generated answers, and ground truth answers for evaluation
        for i, record in enumerate(test_data, 1):
            print(f"Entry #{i}")
            logger.info(f"Processing query {i}/{len(test_data)}")
            
            # Run the query through the pipeline
            result = self.run_query(record["question"])
            print(f"\nQuestion: {record['question']}")       # Print the question for reference
            print(f"Ground Truth: {record['answer']}")     # Print the ground truth answer for reference
            print(f"Generated Answer: {result['answer']} \n") # Print the generated answer for reference

            # Collect retrieved docs (keep as list of dicts for eval)
            all_retrieved.append(result['retrieved_docs'])
            
            # Collect predictions and ground truths
            all_predictions.append(result['answer'])
            all_ground_truths.append(record['answer'])
        
        phase1_time = time.time() - phase1_start  # Time for retrieval + generation per-query processing
        logger.info(f"Phase 1 (Retrieval + Generation): {phase1_time:.2f}s")
        
        # ===== PHASE 2: METRIC COMPUTATION =====
        phase2_start = time.time()  # Track time for metric computation
        
        # Evaluating the retrieval process
        retrieval_metrics = evaluate_retrieval(all_retrieved, test_data)
        
        # Evaluating the generation process
        generation_metrics = evaluate_generation(all_predictions, all_ground_truths)
        
        phase2_time = time.time() - phase2_start  # Time for metric computation
        logger.info(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        
        # ===== COMBINE METRICS WITH TIMING =====
        eval_total_time = time.time() - eval_total_start # Total evaluation time (including both phases)
        
        # Combine all metrics and timing into a single dictionary for return and logging
        all_metrics = {
            **retrieval_metrics,
            **generation_metrics,
            'timing': {
                'phase1': phase1_time,       # Retrieval + generation per query
                'phase2': phase2_time,       # Metric computation
                'total': eval_total_time     # Total evaluation time
            }
        }
        
        # Log results
        # logger.info("=== EVALUATION RESULTS ===")
        print(f"\n === EVALUATION RESULTS ===")
        for metric_key, metric_value in all_metrics.items():
            if metric_key != 'timing':
                if isinstance(metric_value, (int, float)):
                    # logger.info(f"{metric_key}: {metric_value:.4f}")
                    print(f"{metric_key}: {metric_value:.4f}")
                else:
                    # logger.info(f"{metric_key}: {metric_value}")
                    print(f"{metric_key}: {metric_value}")
                    
        
        # Log timing
        # logger.info("=== TIMING BREAKDOWN ===")
        print(f"\n === TIMING BREAKDOWN ===")
        # logger.info(f"Phase 1 (Query Processing): {phase1_time:.2f}s")
        # logger.info(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        # logger.info(f"Total Evaluation Time: {eval_total_time:.2f}s")
        print(f"Phase 1 (Query Processing): {phase1_time:.2f}s")
        print(f"Phase 2 (Metric Computation): {phase2_time:.2f}s")
        print(f"Total Evaluation Time: {eval_total_time:.2f}s")
        
        return all_metrics
