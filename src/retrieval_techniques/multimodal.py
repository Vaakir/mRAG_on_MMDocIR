# src/retrieval_techniques/multimodal.py
# Multimodal retrieval: wraps any query technique with visual classification
# and page image retrieval.

import json
import re
from typing import List, Dict, Any

from qdrant_client.http.models import Filter, FieldCondition, MatchValue


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

    def __init__(self, query_technique, embedder, vector_db, generator, max_page_images: int = 2):
        self.query_technique = query_technique
        self.embedder = embedder
        self.vector_db = vector_db
        self.generator = generator
        self.max_page_images = max_page_images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        _trace(f"Question: {question[:120]}")

        # Step 1: Classify
        is_visual = self._classify_visual(question)

        # Step 2: Text retrieval via query technique
        _trace(f"Text retrieval (top_k={top_k})...")
        text_results = self.query_technique.retrieve(question, top_k)
        _trace(f"Got {len(text_results)} text results")

        # Step 3: Add images
        if is_visual:
            _trace(f"VISUAL → image search with original query")
            image_results = self._image_search(question)
            if not image_results:
                _trace(f"No image results → falling back to metadata lookup")
                image_results = self._page_image_lookup(text_results)
        else:
            _trace(f"TEXT → page image lookup from metadata")
            image_results = self._page_image_lookup(text_results)

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
    # Strategy 4 — image search with original query
    # ------------------------------------------------------------------

    def _image_search(self, question: str) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed_query(question)
        results = self.vector_db.retrieve(
            query_emb, top_k=self.max_page_images, allowed_types=["page_image"]
        )
        for r in results:
            p = r.get("payload", {})
            _trace(f"  image: doc={p.get('doc_name')}  page={p.get('page_num')}  score={r.get('score',0):.3f}")
        return results

    # ------------------------------------------------------------------
    # Strategy 2 — page image lookup from text results metadata
    # ------------------------------------------------------------------

    def _page_image_lookup(self, text_results: List[Dict]) -> List[Dict[str, Any]]:
        lookups = self._collect_page_lookups(text_results)
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

    def _collect_page_lookups(self, text_results: List[Dict]) -> List[tuple]:
        """Extract unique (doc_name, page_num) pairs from text results."""
        seen = set()
        lookups = []
        for r in text_results:
            payload = r.get("payload", {})
            doc_stem = payload.get("pdf_name", "").replace(".pdf", "")
            for pn in (payload.get("page_numbers") or []):
                key = (doc_stem, pn)
                if key not in seen:
                    seen.add(key)
                    lookups.append(key)
                    if len(lookups) >= self.max_page_images:
                        return lookups
        return lookups
