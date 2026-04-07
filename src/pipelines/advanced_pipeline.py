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
from data.multimodal_loader import load_page_images, group_images_by_pdf
from preprocessing.chunker import chunk_documents
from preprocessing.image_chunker import page_level as image_page_level, sliding_window as image_sliding_window
from indexing.embedder import TextEmbedder, create_chunk_embeddings, create_image_chunk_embeddings
from indexing.vector_store import QdrantConfig, QdrantVectorDB
from indexing.hybrid_retriever import HybridRetriever
from generation.generator import BaselineGenerator, MultimodalGenerator
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
        self.image_vector_db: Optional[QdrantVectorDB] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.generator: Optional[BaselineGenerator] = None
        self.query_technique = None
        self.chunks: Optional[List[Dict]] = None
        self.image_chunks: Optional[List[Dict]] = None
    
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
    
    def build_image_index(self, force_rebuild: bool = True):
        """
        Build or load the image index for multimodal retrieval.

        Steps:
        1. Load page images from PAGE_IMAGES_DIR
        2. Apply chunking strategy (page_level or sliding_window)
        3. Embed image chunks with Jina CLIP v2
        4. Store in a separate Qdrant collection (IMAGE_COLLECTION)
        """
        if not self.config.PAGE_IMAGES_DIR:
            raise ValueError("PAGE_IMAGES_DIR must be set in config to build image index.")

        page_images_dir = Path(self.config.PAGE_IMAGES_DIR)
        logger.info(f"Building image index from: {page_images_dir}")

        # Reuse the existing vector_db client (local Qdrant only allows one client per path)
        # Just point to a different collection name
        import copy
        self.image_vector_db = copy.copy(self.vector_db)
        self.image_vector_db.config = copy.copy(self.vector_db.config)
        self.image_vector_db.config.collection_name = self.config.IMAGE_COLLECTION
        self.image_vector_db.config.top_k = self.config.IMAGE_TOP_K

        # Check if image collection already exists
        if not force_rebuild:
            existing_count = self.image_vector_db.count_documents()
            if existing_count > 0:
                logger.info(f"Image index already exists with {existing_count} chunks. Skipping rebuild.")
                return

        # Load page image records
        image_records = load_page_images(page_images_dir)
        logger.info(f"Loaded {len(image_records)} page image records")

        # Apply chunking strategy
        strategy = self.config.IMAGE_CHUNKING_STRATEGY
        if strategy == "page_level":
            self.image_chunks = image_page_level(image_records)
        elif strategy == "sliding_window":
            grouped = group_images_by_pdf(image_records)
            self.image_chunks = image_sliding_window(
                grouped,
                window=self.config.IMAGE_SLIDING_WINDOW_SIZE,
                overlap=self.config.IMAGE_SLIDING_WINDOW_OVERLAP,
            )
        else:
            raise ValueError(f"Unknown IMAGE_CHUNKING_STRATEGY: {strategy}")

        logger.info(f"Created {len(self.image_chunks)} image chunks (strategy={strategy})")

        # Embed image chunks
        logger.info("Embedding image chunks...")
        embeddings = create_image_chunk_embeddings(self.image_chunks, self.embedder)
        logger.info(f"Created {len(embeddings)} image embeddings")

        # Index in Qdrant
        self.image_vector_db.create_collection(force_recreate=True)

        image_docs = []
        for i, (chunk, embedding) in enumerate(zip(self.image_chunks, embeddings)):
            image_docs.append({
                'id': i,
                'embedding': embedding,
                'text': '',  # No text for image chunks
                'metadata': {
                    'pdf_name':    chunk['pdf_name'],
                    'page_num':    chunk['page_num'],
                    'page_nums':   chunk['page_nums'],
                    'image_paths': chunk['image_paths'],
                    'chunk_type':  chunk['chunk_type'],
                    'chunk_id':    chunk['chunk_id'],
                }
            })

        self.image_vector_db.index_documents(image_docs)
        logger.info(f"Image index built: {len(image_docs)} chunks in '{self.config.IMAGE_COLLECTION}'")

    def retrieve_images(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve image chunks relevant to a text query via cross-modal search.

        Args:
            question: User's question (text)
            top_k: Number of image chunks to retrieve

        Returns:
            List of image chunk dicts with image_paths and score
        """
        if self.image_vector_db is None:
            raise RuntimeError("Image index not built. Call build_image_index() first.")
        if top_k is None:
            top_k = self.config.IMAGE_TOP_K

        query_embedding = self.embedder.embed_query(question)
        raw_results = self.image_vector_db.retrieve(
            query_embedding=query_embedding,
            collection_name=self.config.IMAGE_COLLECTION,
            top_k=top_k,
        )

        results = []
        for r in raw_results:
            payload = r.get('payload', {})
            results.append({
                'id':          r['id'],
                'score':       r['score'],
                'text':        '',
                'chunk_type':  payload.get('chunk_type', 'image'),
                'chunk_id':    payload.get('chunk_id', ''),
                'pdf_name':    payload.get('pdf_name', ''),
                'page_num':    payload.get('page_num'),
                'page_nums':   payload.get('page_nums', []),
                'image_paths': payload.get('image_paths', []),
            })
        return results

    def initialize_components(self):
        """Initialize generator and query technique."""
        logger.info(f"Initializing generator: {self.config.LLM_MODEL}")
        if self.config.USE_MULTIMODAL_RETRIEVAL:
            self.generator = MultimodalGenerator(self.config.OLLAMA_BASE_URL, self.config.LLM_MODEL)
            logger.info("Using MultimodalGenerator (vision-language model)")
        else:
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
            text_results = self.hybrid_retriever.retrieve(question, top_k=top_k)
        else:
            # Fallback to dense-only
            query_embedding = self.embedder.embed_query(question)
            text_results = self.vector_db.retrieve(query_embedding=query_embedding, top_k=top_k)

        if self.config.USE_MULTIMODAL_RETRIEVAL and self.image_vector_db is not None:
            image_results = self.retrieve_images(question, top_k=self.config.IMAGE_TOP_K)
            return text_results + image_results

        return text_results
    
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

        # Separate text and image chunks
        text_chunks = [r for r in retrieved if r.get('chunk_type', '') not in ('image_page', 'image_window')]
        image_chunks = [r for r in retrieved if r.get('chunk_type', '') in ('image_page', 'image_window')]

        # Format text context
        context = "\n\n".join([
            f"[Document {i+1}]:\n{result['text']}"
            for i, result in enumerate(text_chunks)
        ])

        # Collect all image paths from retrieved image chunks (flattened)
        retrieved_image_paths = [
            path
            for chunk in image_chunks
            for path in chunk.get('image_paths', [])
        ]

        # Generate answer
        if (
            retrieved_image_paths
            and isinstance(self.generator, MultimodalGenerator)
        ):
            answer = self.generator.generate_with_images(
                question, context, retrieved_image_paths
            )
        else:
            answer = self.generator.generate(question, context)

        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "num_docs": len(retrieved),
            "image_paths": retrieved_image_paths,
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
