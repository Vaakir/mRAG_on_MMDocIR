# src/query_techniques/multimodal.py
# Combined Strategy 2 + 4: text-first page-image lookup + visual query reformulation.

from typing import List, Dict, Any

from ..query_techniques.base import QueryTechnique

CLASSIFY_PROMPT = """\
You classify questions for a document retrieval system.
Determine if the question requires VISUAL content (figures, charts, tables, photographs, maps, diagrams) to answer, or if TEXT content alone is sufficient.

Respond with a JSON object only, no other text:
{"visual": true/false, "query": "short image description under 15 words or null"}"""

CLASSIFY_EXAMPLES = """\
Question: "What DNA repair mechanisms does Figure 11 demonstrate?"
{"visual": true, "query": "figure 11 DNA repair mechanisms diagram"}

Question: "What is the total revenue for FY 2021?"
{"visual": false, "query": null}

Question: "How many emojis does the right subfig have compared to the left?"
{"visual": true, "query": "figure with emojis subfigures comparison"}

Question: "What year is printed on the t-shirt the man is wearing?"
{"visual": true, "query": "photograph man wearing t-shirt year printed"}"""


def _trace(msg: str):
    """Print a trace line. Sequential, no interleaving, shows up in notebook output."""
    print(f"  [multimodal] {msg}")


class MultimodalRetrieval(QueryTechnique):
    """
    Multimodal retrieval: routes between text and visual strategies.

    Text path  (Strategy 2): text retrieval → look up page images for
                              retrieved pages by metadata → VLM gets both.
    Visual path (Strategy 4): LLM rewrites query into a short visual
                              description → image retrieval + text retrieval.

    The LLM classifies each question first to decide the path.
    """

    def __init__(self, embedder, retriever, generator, config=None):
        super().__init__(embedder, retriever, generator, config)
        self.vector_db = (config or {}).get("vector_db")
        self.max_page_images = (config or {}).get("max_page_images", 2)
        _trace(f"Initialized — max_page_images={self.max_page_images}, vector_db={'yes' if self.vector_db else 'no'}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print()  # blank line before each query trace
        _trace(f"==== New query ====")
        _trace(f"Question: {question[:120]}")

        # Step 1: Classify
        analysis = self._classify_question(question)
        needs_visual = analysis["needs_visual"]
        visual_query = analysis.get("visual_query")

        # Step 2: Route
        if needs_visual and visual_query and self.vector_db:
            _trace(f"Route → VISUAL PATH (Strategy 4)")
            _trace(f"Visual query: \"{visual_query}\"")
            results = self._visual_retrieval(question, visual_query, top_k)
        else:
            _trace(f"Route → TEXT PATH (Strategy 2)")
            results = self._text_with_page_images(question, top_k)

        # Summary
        n_text = sum(1 for r in results if r.get("payload", {}).get("type", "text") == "text")
        n_img  = sum(1 for r in results if r.get("payload", {}).get("type") == "page_image")
        _trace(f"Done — returning {len(results)} results ({n_text} text + {n_img} page_image)")
        return results

    # ------------------------------------------------------------------
    # Strategy 2 — text-first, page image lookup
    # ------------------------------------------------------------------

    def _text_with_page_images(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve text chunks, then look up page images for those pages."""
        _trace(f"[S2] Text retrieval (top_k={top_k})...")
        text_results = self.retriever.retrieve(question, top_k=top_k)
        _trace(f"[S2] Got {len(text_results)} text chunks")

        for i, r in enumerate(text_results):
            p = r.get("payload", {})
            _trace(f"[S2]   [{i+1}] doc={p.get('pdf_name','?')}  pages={p.get('page_numbers')}  score={r.get('score',0):.3f}")

        if not self.vector_db:
            _trace(f"[S2] No vector_db — skipping page image lookup")
            return text_results

        lookups = self._collect_page_lookups(text_results)
        _trace(f"[S2] Page image lookup for {len(lookups)} pages: {lookups}")
        page_images = self._lookup_page_images(lookups)
        _trace(f"[S2] Found {len(page_images)}/{len(lookups)} page images")

        return text_results + page_images

    # ------------------------------------------------------------------
    # Strategy 4 — visual query reformulation
    # ------------------------------------------------------------------

    def _visual_retrieval(self, question: str, visual_query: str, top_k: int) -> List[Dict[str, Any]]:
        """Reformulated visual query for image search + standard text retrieval."""
        image_budget = min(self.max_page_images, top_k)
        text_budget = max(top_k - image_budget, 2)
        _trace(f"[S4] Budget: {text_budget} text + {image_budget} images")

        # Text retrieval with the original question
        _trace(f"[S4] Text retrieval with original question...")
        text_results = self.retriever.retrieve(question, top_k=text_budget)
        _trace(f"[S4] Got {len(text_results)} text chunks")
        for i, r in enumerate(text_results):
            p = r.get("payload", {})
            _trace(f"[S4]   [{i+1}] doc={p.get('pdf_name','?')}  pages={p.get('page_numbers')}  score={r.get('score',0):.3f}")

        # Image retrieval with the reformulated visual query
        _trace(f"[S4] Image retrieval with visual query: \"{visual_query}\"")
        visual_emb = self.embedder.embed_query(visual_query)
        image_results = self.vector_db.retrieve(
            visual_emb, top_k=image_budget, allowed_types=["page_image"]
        )

        if image_results:
            _trace(f"[S4] Got {len(image_results)} image results:")
            for i, r in enumerate(image_results):
                p = r.get("payload", {})
                _trace(f"[S4]   [{i+1}] doc={p.get('doc_name')}  page={p.get('page_num')}  score={r.get('score',0):.3f}")
        else:
            _trace(f"[S4] No image results — falling back to page lookup from text metadata")
            lookups = self._collect_page_lookups(text_results)
            image_results = self._lookup_page_images(lookups)
            _trace(f"[S4] Fallback found {len(image_results)} page images")

        return text_results + image_results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_question(self, question: str) -> Dict[str, Any]:
        """Ask the LLM whether the question needs visual content."""
        _trace(f"Classifying question...")
        try:
            response = self.generator.chat([
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": CLASSIFY_EXAMPLES},
                {"role": "assistant", "content": "Understood. Send me the question."},
                {"role": "user", "content": question},
            ])
            clean = response.strip().replace("\n", " | ")
            _trace(f"LLM response: {clean}")
            result = self._parse_classification(response)
            _trace(f"Parsed: needs_visual={result['needs_visual']}, visual_query={result.get('visual_query')}")
            return result
        except Exception as e:
            _trace(f"Classification FAILED ({e}) — defaulting to text path")
            return {"needs_visual": False, "visual_query": None}

    @staticmethod
    def _parse_classification(response: str) -> Dict[str, Any]:
        import json, re

        text = response.strip()

        # Extract JSON object from the response (LLM may wrap it in extra text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                needs_visual = bool(data.get("visual", False))
                visual_query = data.get("query")
                if visual_query and str(visual_query).lower() in ("null", "none", "n/a", ""):
                    visual_query = None
                return {"needs_visual": needs_visual, "visual_query": visual_query}
            except json.JSONDecodeError:
                pass

        _trace("Could not parse JSON from LLM response, defaulting to text path")
        return {"needs_visual": False, "visual_query": None}

    # ------------------------------------------------------------------
    # Page image helpers
    # ------------------------------------------------------------------

    def _collect_page_lookups(self, text_results: List[Dict]) -> List[tuple]:
        """Extract unique (doc_name, page_num) pairs from text results."""
        seen = set()
        lookups = []
        for r in text_results:
            payload = r.get("payload", {})
            pdf_name = payload.get("pdf_name", "")
            doc_stem = pdf_name.replace(".pdf", "")
            page_nums = payload.get("page_numbers") or []

            for pn in page_nums:
                key = (doc_stem, pn)
                if key not in seen:
                    seen.add(key)
                    lookups.append(key)
                    if len(lookups) >= self.max_page_images:
                        return lookups
        return lookups

    def _lookup_page_images(self, lookups: List[tuple]) -> List[Dict[str, Any]]:
        """Fetch page images from Qdrant by (doc_name, page_num) metadata."""
        if not lookups or not self.vector_db:
            return []

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        results = []
        collection = self.vector_db.config.VECTOR_DB_COLLECTION

        for doc_name, page_num in lookups:
            try:
                page_filter = Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value="page_image")),
                    FieldCondition(key="doc_name", match=MatchValue(value=doc_name)),
                    FieldCondition(key="page_num", match=MatchValue(value=page_num)),
                ])

                points, _ = self.vector_db.client.scroll(
                    collection_name=collection,
                    scroll_filter=page_filter,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )

                if points:
                    pt = points[0]
                    results.append({
                        "id": pt.id,
                        "score": 0.0,
                        "payload": pt.payload,
                        "text": pt.payload.get("text", ""),
                    })
                    _trace(f"  Found page image: {doc_name} page {page_num}")
                else:
                    _trace(f"  Not found: {doc_name} page {page_num}")
            except Exception as e:
                _trace(f"  Error looking up {doc_name} p{page_num}: {e}")

        return results
