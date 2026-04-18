# src/retrieval_techniques/multimodal.py
# Multimodal retrieval: wraps any query technique with visual classification
# and page image retrieval.

import json
import re
from typing import List, Dict, Any, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny


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
    Wraps any query technique with visual classification and page image retrieval.

    Flow:
        1. LLM classifies the question as visual or text.
        2. Run the configured query technique for text retrieval.
        3. Add page images:
           - VISUAL → image search with the original query (Strategy 4)
           - TEXT   → look up page images from text results metadata (Strategy 2)
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

        # Step 1: Classify (or use forced routing from evaluation)
        is_visual = self._classify_visual(question) if is_visual is None else is_visual

        # Decide split per policy
        if is_visual:
            text_k, image_k = self._split_text_image_k(top_k, self.text_ratio_visual)
        else:
            text_k, image_k = self._split_text_image_k(top_k, self.text_ratio_non_visual)

        # Step 2: Text retrieval
        # Use provided technique (from agent) or fall back to default
        used_technique = technique if technique is not None else self.query_technique
        _trace(f"Text retrieval (text_k={text_k})...")
        text_results = used_technique.retrieve(question, text_k)
        _trace(f"Got {len(text_results)} text results")

        # Step 3: Image retrieval if we have budget
        if image_k <= 0:
            _trace(f"Returning {len(text_results)} text + 0 images")
            return text_results
        doc_names = self._extract_doc_names(text_results)

        # Strategy 4 first: doc-filtered CLIP search (finds visually relevant images)
        if is_visual:
            _trace(f"Strategy 4 → doc-filtered image search (image_k={image_k}, docs={doc_names[:3]})")
            image_results = self._image_search(question, top_k=image_k, doc_names=doc_names)
        else:
            image_results = []

        # Fallback to Strategy 2: page lookup from text chunk metadata
        if len(image_results) < image_k:
            remaining = image_k - len(image_results)
            _trace(f"Strategy 2 fallback → page image lookup (remaining={remaining})")
            seen_ids = {r["id"] for r in image_results}
            for r in self._page_image_lookup(text_results, limit=remaining):
                if r["id"] not in seen_ids:
                    image_results.append(r)
                    seen_ids.add(r["id"])

        _trace(f"Returning {len(text_results)} text + {len(image_results)} images")
        return text_results + image_results
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
            # text chunks store pdf_name; image chunks store doc_name
            name = p.get("pdf_name", p.get("doc_name", ""))
            name = name.replace(".pdf", "")
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
        k = int(top_k) if top_k is not None else int(self.max_page_images)
        k = max(1, min(k, int(self.max_page_images)))

        # Build filter: image types AND (optionally) constrain to specific documents
        image_types = ["page_image", "figure", "evidence"]
        must_conditions = [
            FieldCondition(key="type", match=MatchAny(any=image_types))
        ]
        if doc_names:
            must_conditions.append(
                FieldCondition(key="doc_name", match=MatchAny(any=doc_names))
            )

        from qdrant_client.http.models import Filter as QFilter
        query_filter = QFilter(must=must_conditions)

        collection = self.vector_db.config.VECTOR_DB_COLLECTION
        try:
            import numpy as np
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
    # Strategy 2 — page image lookup from text results metadata
    # ------------------------------------------------------------------

    def _page_image_lookup(self, text_results: List[Dict], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        lookups = self._collect_page_lookups(text_results, limit=limit)
        if not lookups:
            return []

        _trace(f"  Looking up {len(lookups)} pages: {lookups}")
        results = []
        collection = self.vector_db.config.VECTOR_DB_COLLECTION

        for doc_name, page_num in lookups:
            try:
                points, _ = self.vector_db.client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="type", match=MatchValue(value="page_image")),
                        FieldCondition(key="doc_name", match=MatchValue(value=doc_name)),
                        FieldCondition(key="page_num", match=MatchValue(value=page_num)),
                    ]),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if points:
                    pt = points[0]
                    results.append({
                        "id": pt.id, "score": 0.0,
                        "payload": pt.payload,
                        "text": pt.payload.get("text", ""),
                    })
                    _trace(f"  Found: {doc_name} page {page_num}")
                else:
                    _trace(f"  Not found: {doc_name} page {page_num}")
            except Exception as e:
                _trace(f"  Error: {doc_name} p{page_num}: {e}")

        return results

    def _collect_page_lookups(self, text_results: List[Dict], limit: Optional[int] = None) -> List[tuple]:
        """Extract unique (doc_name, page_num) pairs from text results."""
        seen = set()
        lookups = []
        for r in text_results:
            payload = r.get("payload", {})
            doc_stem = payload.get("pdf_name", "").replace(".pdf", "")
            cap = int(limit) if limit is not None else int(self.max_page_images)
            for pn in (payload.get("page_numbers") or []):
                key = (doc_stem, pn)
                if key not in seen:
                    seen.add(key)
                    lookups.append(key)
                    if len(lookups) >= cap:
                        return lookups
        return lookups
    
    def _split_text_image_k(self, total_k: int, text_ratio: float) -> tuple[int, int]:
        """Compute (text_k, image_k) from a total budget.

        - Guarantees at least 1 text.
        - image_k is capped by self.max_page_images.
        """
        total_k = max(1, int(total_k))
        text_k = int(round(total_k * float(text_ratio)))
        text_k = max(1, min(text_k, total_k))
        image_k = total_k - text_k
        image_k = max(0, min(image_k, int(self.max_page_images)))
        return text_k, image_k
