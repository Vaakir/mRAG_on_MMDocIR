# src/pipelines/advanced_pipeline.py
# Advanced RAG pipeline with multimodal retrieval (text + page images).
# Uses any query technique + MultimodalRetriever for image-aware retrieval.

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from generation.prompts import get_prompt_strategy
from config.config import AdvancedConfig
from pipelines.base_pipeline import BaseRAGPipeline
from preprocessing.image_processor import load_page_image_chunks, load_figure_chunks, load_evidence_chunks
from generation.generator import VisionGenerator
from query_techniques import get_query_technique
from retrieval_techniques import MultimodalRetriever

logger = logging.getLogger(__name__)


class AdvancedRAGPipeline(BaseRAGPipeline):
    """
    Advanced RAG pipeline.

    - Text + page images in one Qdrant collection (same CLIP embedding space).
    - Any query technique (standard, hyde, multi_query, etc.) for text retrieval.
    - MultimodalRetriever wraps the query technique with visual classification
      and page image retrieval (Strategy 2 + 4).
    - Routes generation to VLM when image chunks are in the results.
    """

    def __init__(self, config: AdvancedConfig):
        super().__init__(config)
        self.query_technique = None
        self.prompt_strategy = None
        
        self.multimodal_retriever: Optional[MultimodalRetriever] = None
        self.vlm: Optional[VisionGenerator] = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = True):
        """Build multimodal index: text chunks + page images."""
        logger.info(f"Building multimodal index with {type(self.config).__name__}...")
        start_time = time.time()

        # Init embedder + Qdrant
        from indexing.embedder import TextEmbedder
        from indexing.vector_database import QdrantVectorDB

        logger.info(f"Initialising embedder: {self.config.EMBEDDING_MODEL}")
        self.embedder = TextEmbedder(self.config.EMBEDDING_MODEL)
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

        from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
        chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
        self.chunks = load_preprocessed_chunks(chunks_path)
        print_chunk_statistics(self.chunks)

        text_embeddings = self.embedder.embed_texts(
            [c["text"] for c in self.chunks],
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
        )

        # Page images
        page_chunks = []
        page_embeddings = None
        if self.config.USE_MULTIMODAL:
            page_chunks = load_page_image_chunks(
                self.config.PAGE_IMAGES_TRAIN_DIR, self.config.DATA_DIR,
            )
            if page_chunks:
                abs_paths = [str(self.config.DATA_DIR / c["image_path"]) for c in page_chunks]
                page_embeddings = self.embedder.embed_images(abs_paths)

        # Extracted figures (pictures, charts, tables)
        figure_chunks = []
        figure_embeddings = None
        if self.config.USE_MULTIMODAL and hasattr(self.config, "FIGURES_TRAIN_DIR"):
            figure_chunks = load_figure_chunks(
                self.config.FIGURES_TRAIN_DIR, self.config.DATA_DIR,
            )
            if figure_chunks:
                abs_paths = [str(self.config.DATA_DIR / c["image_path"]) for c in figure_chunks]
                figure_embeddings = self.embedder.embed_images(abs_paths)

                # Evidence crops
        evidence_chunks = []
        evidence_embeddings = None
        if self.config.USE_MULTIMODAL and hasattr(self.config, "IMAGES_TRAIN_DIR"):
            evidence_chunks = load_evidence_chunks(
                self.config.IMAGES_TRAIN_DIR, self.config.DATA_DIR
            )
            if evidence_chunks:
                abs_paths = []
                for c in evidence_chunks:
                    img_val = c.get("image_path") or c.get("image_paths")
                    if isinstance(img_val, list):
                        abs_paths.append(str(self.config.DATA_DIR / img_val[0]))
                    else:
                        abs_paths.append(str(self.config.DATA_DIR / img_val))
                evidence_embeddings = self.embedder.embed_images(abs_paths)

        # Index into Qdrant
        self.vector_db.create_collection(force_recreate=True)
        qdrant_docs = []
        doc_id = 0

        for chunk, emb in zip(self.chunks, text_embeddings):
            qdrant_docs.append({
                "id": doc_id, "embedding": emb, "text": chunk["text"],
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

        if page_embeddings is not None:
            for chunk, emb in zip(page_chunks, page_embeddings):
                qdrant_docs.append({
                    "id": doc_id, "embedding": emb, "text": chunk["text"],
                    "metadata": {
                        "type": "page_image",
                        "image_path": chunk["image_path"],
                        "doc_name": chunk["doc_name"],
                        "page_num": chunk["page_num"],
                    },
                })
                doc_id += 1

        if figure_embeddings is not None:
            for chunk, emb in zip(figure_chunks, figure_embeddings):
                qdrant_docs.append({
                    "id": doc_id, "embedding": emb, "text": chunk["text"],
                    "metadata": {
                        "type": "figure",
                        "image_path": chunk["image_path"],
                        "doc_name": chunk["doc_name"],
                        "page_num": chunk["page_num"],
                        "label": chunk["label"],
                    },
                })
                doc_id += 1
        
        if evidence_embeddings is not None:
            for chunk, emb in zip(evidence_chunks, evidence_embeddings):
                qdrant_docs.append({
                    "id": doc_id, "embedding": emb, "text": chunk["text"],
                    "metadata": {
                        "type": chunk.get("type", "evidence"),
                        "image_path": chunk.get("image_path") or chunk.get("image_paths"),
                        "question_id": chunk.get("question_id"),
                        "doc_name": chunk.get("doc_name")
                    },
                })
                doc_id += 1

        self.vector_db.index_documents(qdrant_docs)
        logger.info(f"Indexed {doc_id} docs ({len(self.chunks)} text, {len(page_chunks)} page images, {len(figure_chunks)} figures, {len(evidence_chunks)} evidence) in {time.time() - start_time:.2f}s")
        self._initialize_retriever()

    def _load_chunks_for_bm25(self):
        if self.config.USE_HYBRID_RETRIEVAL:
            from data.chunk_loader import load_preprocessed_chunks
            self.chunks = load_preprocessed_chunks(Path(self.config.PREPROCESSED_CHUNKS_FILE))

    # ------------------------------------------------------------------
    # Component initialisation
    # ------------------------------------------------------------------

    def initialize_components(self):
        """Init generator (VLM), prompt strategy, query technique, and multimodal retriever."""
        # Use VisionGenerator instead of base BaselineGenerator
        logger.info(f"Initialising generator: {self.config.LLM_MODEL}")
        self.generator = VisionGenerator(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.LLM_MODEL,
            api_key=self.config.OLLAMA_API_KEY,
            config=self.config,
            temperature=self.config.LLM_TEMPERATURE,
            top_p=self.config.LLM_TOP_P,
            max_tokens=self.config.LLM_MAX_TOKENS,
            max_retries=self.config.LLM_MAX_RETRIES,
            retry_delay=self.config.LLM_RETRY_DELAY
        )
        self.vlm = self.generator

        # Prompt strategy
        logger.info(f"Initializing prompting strategy: {self.config.PROMPTING_STRATEGY}")
        strategy_config = self.config.PROMPTING_STRATEGY_CONFIG.copy()
        if (self.config.PROMPTING_STRATEGY == 'ensemble' and
            strategy_config.get('aggregation_method') == 'embedding_similarity'):
            strategy_config['embedder'] = self.embedder
        self.prompt_strategy = get_prompt_strategy(
            self.config.PROMPTING_STRATEGY,
            self.generator,
            strategy_config,
        )

        # Query technique
        logger.info(f"Initialising query technique: {self.config.QUERY_TECHNIQUE}")
        self.query_technique = get_query_technique(
            self.config.QUERY_TECHNIQUE,
            self.embedder,
            self.hybrid_retriever or getattr(self, "retriever", None),
            self.generator,
            self.config.QUERY_TECHNIQUE_CONFIG,
        )

        # Multimodal retriever
        if self.config.USE_MULTIMODAL:
            logger.info("Initialising multimodal retriever...")
            self.multimodal_retriever = MultimodalRetriever(
                query_technique=self.query_technique,
                embedder=self.embedder,
                vector_db=self.vector_db,
                generator=self.generator,
                max_page_images=self.config.QUERY_TECHNIQUE_CONFIG.get("max_page_images", 1),
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_with_technique(self, question: str, top_k: Optional[int] = None, is_visual: Optional[bool] = None, doc_name: Optional[str] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.TOP_K
        query = f"[Document: {doc_name}] {question}" if doc_name else question
        if self.multimodal_retriever:
            return self.multimodal_retriever.retrieve(query, top_k, is_visual=is_visual)
        if self.query_technique:
            return self.query_technique.retrieve(query, top_k)
        raise RuntimeError("Call initialize_components() first.")

    # ------------------------------------------------------------------
    # Generation routing
    # ------------------------------------------------------------------

    def _generate(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        """Route to VLM if page images or figures were retrieved, otherwise prompt strategy."""
        image_types = {"page_image", "figure", "evidence"}
        image_results = [r for r in retrieved if r.get("payload", {}).get("type") in image_types]
        text_results  = [r for r in retrieved if r.get("payload", {}).get("type") not in image_types]

        if image_results and self.vlm is not None:
            image_paths = []
            for r in image_results:
                ip = r["payload"].get("image_path", [])
                if not isinstance(ip, list):
                    ip = [ip]
                for p in ip:
                    image_paths.append(str(self.config.DATA_DIR / p))
                    if len(image_paths) >= self.config.MAX_VLM_IMAGES:
                        break
                if len(image_paths) >= self.config.MAX_VLM_IMAGES:
                    break

            text_context = "\n\n".join(
                f"[Document {i+1}]:\n{r['text']}" for i, r in enumerate(text_results)
            )

            # VLM generation through prompt strategy (handles system prompt + post-processing)
            try:
                if self.prompt_strategy:
                    answer = self.prompt_strategy.generate_with_images(question, image_paths, text_context)
                else:
                    answer = self.vlm.generate_with_images(question, image_paths, text_context)
                if not answer.startswith("Error generating response:"):
                    return answer
                logger.warning(f"VLM returned error, falling back to text-only generation")
            except Exception as e:
                logger.warning(f"VLM failed ({e}), falling back to text-only generation")

        # Text-only fallback — through prompt strategy
        context = "\n\n".join(
            f"[Document {i+1}]:\n{r['text']}" for i, r in enumerate(retrieved)
        )
        if self.prompt_strategy:
            return self.prompt_strategy.generate(question, context)
        return self.generator.generate(question, context)

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def run_query(self, question: str, use_technique: bool = True, top_k: Optional[int] = None, is_visual: Optional[bool] = None, doc_name: Optional[str] = None, record: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Run a single query through the advanced pipeline retrieval and generation.
        """
        top_k = top_k or self.config.TOP_K

        # If a full record is passed (e.g. from evaluate), extract advanced properties
        if record and is_visual is None:
            types = record.get("types") or []
            is_visual = any(t in {"Figure", "Chart", "Table"} for t in types)
            if not types:
                q = record.get("question", "").lower()
                is_visual = any(k in q for k in ["logo", "color", "colour", "shown", "figure", "chart", "table"])
        
        if record and doc_name is None:
            doc_name = Path(record["pdf_path"]).stem if record.get("pdf_path") else None

        if use_technique:
            retrieved = self.retrieve_with_technique(question, top_k, is_visual=is_visual, doc_name=doc_name)
        else:
            retrieved = self.retrieve(question, top_k)

        answer = self._generate(question, retrieved)
        return {"question": question, "retrieved_docs": retrieved, "answer": answer, "num_docs": len(retrieved)}
