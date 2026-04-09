# src/pipelines/advanced_pipeline.py
# Advanced RAG pipeline with multimodal retrieval (text + page images + evidence crops).
# Baseline pipeline = text only.
# Advanced pipeline = text + image in one Qdrant collection, routes to VLM when images retrieved.

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config import AdvancedConfig
from pipelines.base_pipeline import BaseRAGPipeline
from preprocessing.image_processor import load_page_image_chunks, load_evidence_chunks
from indexing.embedder import TextEmbedder
from generation.generator import VisionGenerator
from query_techniques import get_query_technique
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation

logger = logging.getLogger(__name__)


class AdvancedRAGPipeline(BaseRAGPipeline):
    """
    Advanced RAG pipeline.
    Same vector space for text chunks, page images, and evidence crops.
    Routes generation to qwen3-vl when image chunks are in the top-K results.
    """

    def __init__(self, config: AdvancedConfig):
        super().__init__(config)
        self.query_technique = None
        self.vlm: Optional[VisionGenerator] = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = True):
        """
        Build the multimodal index:
          1. Text chunks  (from preprocessed JSON, text encoder)
          2. Page images  (page_images_train, image encoder)
          3. Evidence crops (train.jsonl question texts, text encoder → image path in metadata)
        All stored in one Qdrant collection.
        """
        logger.info(f"Building multimodal index with {type(self.config).__name__}...")
        start_time = time.time()

        # Step 1: Init embedder
        logger.info(f"Initialising embedder: {self.config.EMBEDDING_MODEL}")
        self.embedder = TextEmbedder(self.config.EMBEDDING_MODEL)

        # Step 2: Init Qdrant
        from indexing.vector_database import QdrantVectorDB
        self.vector_db = QdrantVectorDB(self.config)

        # Check existing collection
        collection_exists = False
        try:
            count = self.vector_db.count_documents()
            collection_exists = count > 0
            logger.info(f"Existing collection has {count} documents")
        except Exception:
            pass

        if not force_rebuild and collection_exists:
            logger.info("Loading existing multimodal index...")
            self._load_chunks_for_bm25()
            self._initialize_retriever()
            logger.info(f"Index loaded in {time.time() - start_time:.2f}s")
            return

        # ---------- Build from scratch ----------
        logger.info("Building new multimodal index from scratch...")

        # --- Text chunks ---
        from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
        chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
        logger.info(f"Loading text chunks from {chunks_path}")
        self.chunks = load_preprocessed_chunks(chunks_path)
        print_chunk_statistics(self.chunks)
        logger.info(f"Loaded {len(self.chunks)} text chunks")

        text_embeddings = self.embedder.embed_texts(
            [c["text"] for c in self.chunks],
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
        )

        # --- Page images ---
        page_chunks = []
        page_embeddings = None
        if self.config.USE_MULTIMODAL:
            page_chunks = load_page_image_chunks(
                self.config.PAGE_IMAGES_TRAIN_DIR,
                self.config.DATA_DIR,
            )
            if page_chunks:
                # Resolve relative paths to absolute for embedding (need to open files)
                abs_paths = [str(self.config.DATA_DIR / c["image_path"]) for c in page_chunks]
                page_embeddings = self.embedder.embed_images(abs_paths)

        # --- Evidence crops ---
        evidence_chunks = []
        evidence_embeddings = None
        if self.config.USE_MULTIMODAL:
            evidence_chunks = load_evidence_chunks(
                self.config.IMAGES_TRAIN_DIR,
                self.config.TRAIN_JSONL,
                self.config.DATA_DIR,
            )
            if evidence_chunks:
                evidence_embeddings = self.embedder.embed_texts(
                    [c["text"] for c in evidence_chunks],  # question text
                    batch_size=self.config.EMBEDDING_BATCH_SIZE,
                )

        # --- Index into Qdrant ---
        self.vector_db.create_collection(force_recreate=True)

        qdrant_docs = []
        doc_id = 0

        # Text chunks
        for chunk, emb in zip(self.chunks, text_embeddings):
            qdrant_docs.append({
                "id": doc_id,
                "embedding": emb,
                "text": chunk["text"],
                "metadata": {
                    "type": "text",
                    "pdf_name": str(chunk.get("pdf_name", "unknown")),
                    "pdf_path": str(chunk.get("pdf_path", "unknown")),
                    "chunk_id": chunk.get("chunk_id"),
                    "char_len": int(chunk.get("char_len", 0)),
                    "page_numbers": chunk.get("page_numbers"),
                },
            })
            doc_id += 1

        # Page images
        if page_embeddings is not None:
            for chunk, emb in zip(page_chunks, page_embeddings):
                qdrant_docs.append({
                    "id": doc_id,
                    "embedding": emb,
                    "text": chunk["text"],
                    "metadata": {
                        "type": "page_image",
                        "image_path": chunk["image_path"],
                        "doc_name": chunk["doc_name"],
                        "page_num": chunk["page_num"],
                    },
                })
                doc_id += 1

        # Evidence crops
        if evidence_embeddings is not None:
            for chunk, emb in zip(evidence_chunks, evidence_embeddings):
                qdrant_docs.append({
                    "id": doc_id,
                    "embedding": emb,
                    "text": chunk["text"],   # question text (display/BM25)
                    "metadata": {
                        "type": "evidence",
                        "image_path": chunk["image_path"],  # list[str]
                        "question_id": chunk.get("question_id"),
                        "doc_name": chunk.get("doc_name", ""),
                    },
                })
                doc_id += 1

        self.vector_db.index_documents(qdrant_docs)
        elapsed = time.time() - start_time
        logger.info(
            f"Indexed {doc_id} documents "
            f"({len(self.chunks)} text, {len(page_chunks)} page images, "
            f"{len(evidence_chunks)} evidence crops) in {elapsed:.2f}s"
        )

        self._initialize_retriever()

    def _load_chunks_for_bm25(self):
        """Load text chunks into memory for the BM25 component."""
        if self.config.USE_HYBRID_RETRIEVAL:
            from data.chunk_loader import load_preprocessed_chunks
            chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
            self.chunks = load_preprocessed_chunks(chunks_path)

    # ------------------------------------------------------------------
    # Component initialisation
    # ------------------------------------------------------------------

    def initialize_components(self):
        """Init single VisionGenerator for both text and image, and query technique."""
        # Use VisionGenerator for everything — one model on server, no swapping
        logger.info(f"Initialising generator (VisionGenerator): {self.config.LLM_MODEL}")
        self.generator = VisionGenerator(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.LLM_MODEL,
        )
        self.vlm = self.generator  # same instance — text uses think=True, images think=False

        logger.info(f"Initialising query technique: {self.config.QUERY_TECHNIQUE}")
        self.query_technique = get_query_technique(
            self.config.QUERY_TECHNIQUE,
            self.embedder,
            self.hybrid_retriever or getattr(self, "retriever", None),
            self.generator,
            self.config.QUERY_TECHNIQUE_CONFIG,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_with_technique(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.TOP_K
        if self.query_technique is None:
            raise RuntimeError("Call initialize_components() first.")
        return self.query_technique.retrieve(question, top_k)

    # ------------------------------------------------------------------
    # Generation routing
    # ------------------------------------------------------------------

    def _generate(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Route to VLM if any image chunks were retrieved, otherwise use text LLM.
        BM25-only results have no 'type' in payload — treat those as text.
        """
        _IMAGE_TYPES = {"page_image", "evidence"}
        image_results = [r for r in retrieved if r.get("payload", {}).get("type") in _IMAGE_TYPES]
        text_results  = [r for r in retrieved if r.get("payload", {}).get("type") not in _IMAGE_TYPES]

        if image_results and self.vlm is not None:
            # Collect image paths — stored as relative to DATA_DIR, resolve now
            image_paths = []
            for r in image_results:
                ip = r["payload"].get("image_path", [])
                if not isinstance(ip, list):
                    ip = [ip]
                for p in ip:
                    image_paths.append(str(self.config.DATA_DIR / p))

            text_context = "\n\n".join(
                f"[Document {i+1}]:\n{r['text']}"
                for i, r in enumerate(text_results)
            )
            return self.vlm.generate_with_images(question, image_paths, text_context)

        # Text-only fallback
        context = "\n\n".join(
            f"[Document {i+1}]:\n{r['text']}"
            for i, r in enumerate(retrieved)
        )
        return self.generator.generate(question, context)

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def run_query(self, question: str, use_technique: bool = True, top_k: Optional[int] = None) -> Dict[str, Any]:
        top_k = top_k or self.config.TOP_K

        if use_technique and self.query_technique:
            retrieved = self.retrieve_with_technique(question, top_k)
        else:
            retrieved = self.retrieve(question, top_k)

        answer = self._generate(question, retrieved)

        return {
            "question": question,
            "retrieved_docs": retrieved,
            "answer": answer,
            "num_docs": len(retrieved),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_questions: List[Dict[str, Any]], use_technique: bool = True):
        """
        Two-phase evaluation (retrieval parallel → generation parallel).
        Same structure as before but generation routes through _generate().
        """
        eval_start = time.time()
        logger.info(f"Evaluating {len(test_questions)} questions (technique={use_technique})...")

        test_subset = test_questions[:self.config.EVAL_SUBSET_SIZE]

        # Phase 1: Retrieval
        phase1_start = time.time()
        all_retrievals = [None] * len(test_subset)

        with ThreadPoolExecutor(max_workers=self.config.RETRIEVAL_WORKERS) as executor:
            futures = {}
            for i, tq in enumerate(test_subset):
                fn = self.retrieve_with_technique if (use_technique and self.query_technique) else self.retrieve
                futures[executor.submit(fn, tq["question"], self.config.TOP_K)] = i

            completed = 0
            for future in as_completed(futures):
                all_retrievals[futures[future]] = future.result()
                completed += 1
                logger.info(f"  Retrieved {completed}/{len(test_subset)}")

        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1 done in {phase1_time:.2f}s")

        # Phase 2: Generation
        phase2_start = time.time()
        generation_results = []

        with ThreadPoolExecutor(max_workers=self.config.GENERATION_WORKERS) as executor:
            futures = [
                (i, tq, executor.submit(self._generate, tq["question"], all_retrievals[i]))
                for i, tq in enumerate(test_subset)
            ]

            completed_count = 0
            for i, tq, future in futures:
                answer = future.result()
                completed_count += 1
                logger.info(f"  Generated {completed_count}/{len(futures)}")

                print(f"\n{'-'*80}")
                print(f"Question: {tq['question']}")
                print(f"Ground Truth: {tq['answer']}")
                print(f"Generated Answer: {answer}")

                generation_results.append({
                    "question": tq["question"],
                    "answer": answer,
                    "expected_answer": tq.get("answer", ""),
                })

        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 done in {phase2_time:.2f}s")

        retrieval_metrics = evaluate_retrieval(
            [r for r in all_retrievals],
            test_subset,
        )
        generation_metrics = evaluate_generation(
            [r["answer"] for r in generation_results],
            [r["expected_answer"] for r in generation_results],
        )

        total_time = time.time() - eval_start
        logger.info(f"Total evaluation time: {total_time:.2f}s")
        logger.info(f"Retrieval metrics: {retrieval_metrics}")
        logger.info(f"Generation metrics: {generation_metrics}")

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "timing": {"total": total_time, "phase1": phase1_time, "phase2": phase2_time},
        }
