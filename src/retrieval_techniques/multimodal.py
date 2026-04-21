# src/retrieval_techniques/multimodal.py
# Multimodal retrieval: wraps any query technique with visual classification
# and image retrieval (Strategy 4 — doc-filtered CLIP search).

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchAny


# ----- Prompts for visual classification -----

CLASSIFY_PROMPT = """\
You classify questions for a document retrieval system.
Determine if the question requires VISUAL content (figures, charts, tables, photographs, maps, diagrams) to answer, or if TEXT content alone is sufficient.

Respond with a JSON object only, no other text:
{"visual": true/false}"""

CLASSIFY_EXAMPLES = """\
Question: "What DNA repair mechanisms does Figure 11 demonstrate?"
{"visual": true}

Question: "What is the total revenue for FY 2021?"
{"visual": false}

Question: "How many emojis does the right subfig have compared to the left?"
{"visual": true}

Question: "What year is printed on the t-shirt the man is wearing?"
{"visual": true}"""


def _trace(msg: str):
    """Sequential print trace for notebook readability."""
    print(f"  [retrieval] {msg}")


# ----- Main class -----

class MultimodalRetriever:
    """
    Wraps any query technique with visual classification and image retrieval.

    Flow:
        1. LLM classifies the question as visual or text.
        2. Run the configured query technique for text retrieval.
        3. If visual: doc-filtered CLIP image search (Strategy 4).
    """

    def __init__(self, query_technique, embedder, vector_db, generator, max_page_images: int = 1):
        self.query_technique = query_technique
        self.embedder = embedder
        self.vector_db = vector_db
        self.generator = generator
        self.max_page_images = max_page_images

        self.text_ratio_non_visual = 1.0
        self.text_ratio_visual = 0.8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5, is_visual: Optional[bool] = None, technique=None) -> List[Dict[str, Any]]:
        _trace(f"Question: {question[:120]}")

        is_visual = self._classify_visual(question) if is_visual is None else is_visual

        if is_visual:
            text_k, image_k = self._split_text_image_k(top_k, self.text_ratio_visual)
        else:
            text_k, image_k = self._split_text_image_k(top_k, self.text_ratio_non_visual)

        _trace(f"Text retrieval (text_k={text_k})...")
        # Use provided technique or fall back to the initialized one
        query_technique = technique if technique is not None else self.query_technique
        text_results = query_technique.retrieve(question, text_k)
        _trace(f"Got {len(text_results)} text results")

        if image_k <= 0:
            _trace(f"Returning {len(text_results)} text + 0 images")
            return text_results

        doc_names = self._extract_doc_names(text_results)
        _trace(f"Strategy 4 → doc-filtered image search (image_k={image_k}, docs={doc_names[:3]})")
        image_results = self._image_search(question, top_k=image_k, doc_names=doc_names)

        combined = self._merge_by_cosine_score(text_results, image_results)
        _trace(f"Returning {len(text_results)} text + {len(image_results)} images (cosine-merged)")
        return combined

    # ------------------------------------------------------------------
    # Visual classification
    # ------------------------------------------------------------------

    def _classify_visual(self, question: str) -> bool:
        _trace("Classifying: visual or text?")
        try:
            response = self.generator.chat([
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": CLASSIFY_EXAMPLES},
                {"role": "assistant", "content": "Understood. Send me the question."},
                {"role": "user", "content": question},
            ])
            _trace(f"LLM: {response.strip()}")

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                is_visual = bool(data.get("visual", False))
                _trace(f"→ {'VISUAL' if is_visual else 'TEXT'}")
                return is_visual
        except Exception as e:
            _trace(f"Classification failed ({e}), defaulting to TEXT")
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_doc_names(text_results: List[Dict]) -> List[str]:
        """Extract unique document names from text retrieval results."""
        seen = set()
        names = []
        for r in text_results:
            p = r.get("payload", {})
            name = p.get("pdf_name", p.get("doc_name", "")).replace(".pdf", "")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Strategy 4 — image search with original query (doc-filtered)
    # ------------------------------------------------------------------

    def _image_search(self, question: str, top_k: Optional[int] = None,
                      doc_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed_query(question)
        k = max(1, min(int(top_k) if top_k is not None else self.max_page_images, self.max_page_images))

        must_conditions = [
            FieldCondition(key="type", match=MatchAny(any=["page_image", "figure", "evidence"]))
        ]
        if doc_names:
            must_conditions.append(
                FieldCondition(key="doc_name", match=MatchAny(any=doc_names))
            )

        query_filter = Filter(must=must_conditions)
        collection = self.vector_db.config.VECTOR_DB_COLLECTION

        try:
            qe = query_emb.tolist() if isinstance(query_emb, np.ndarray) else query_emb
            search_results = self.vector_db.client.query_points(
                collection_name=collection,
                query=qe,
                limit=k,
                query_filter=query_filter,
            ).points
        except Exception as e:
            _trace(f"  image search error: {e}")
            return []

        results = []
        for pt in search_results:
            results.append({
                "id": pt.id,
                "score": pt.score,
                "payload": pt.payload,
                "text": pt.payload.get("text", ""),
            })
            p = pt.payload
            _trace(f"  image: doc={p.get('doc_name')}  page={p.get('page_num')}  score={pt.score:.3f}")
        return results

    # ------------------------------------------------------------------
    # Score normalisation and merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_by_cosine_score(
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Insert image results into the text ranking at the position where their
        cosine score fits among text dense_scores. Text RRF order is preserved —
        only the image insertion point is determined by cosine similarity.
        BM25-only hits (dense_score=0.0) stay in their RRF position.
        """
        merged = list(text_results)
        for image in image_results:
            image_score = image["score"]
            insert_at = next(
                (i for i, r in enumerate(merged) if r.get("dense_score", 0.0) < image_score),
                len(merged)
            )
            merged.insert(insert_at, image)
        return merged

    # ------------------------------------------------------------------
    # Budget split
    # ------------------------------------------------------------------

    def _split_text_image_k(self, total_k: int, text_ratio: float) -> tuple[int, int]:
        """Compute (text_k, image_k) from a total budget."""
        total_k = max(1, int(total_k))
        text_k = max(1, min(int(round(total_k * float(text_ratio))), total_k))
        image_k = max(0, min(total_k - text_k, self.max_page_images))
        return text_k, image_k