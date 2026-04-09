# DAT560 mRAG Project — Report Template

**Deadline: April 20, 2026**

---

## 1. Introduction

- Motivation: challenges of retrieval over multimodal documents (text + images in PDFs)
- Research questions, for example:
  - Does multimodal retrieval with Jina CLIP v2 image embeddings outperform text-only retrieval?
  - Which query technique maximises Precision@5?
- Brief overview of the three systems built
- Paper structure summary

---

## 2. Background & Related Work

- RAG fundamentals (Lewis et al. 2020)
- Multimodal RAG and MMDocIR dataset description (collection size, modalities, splits)
- Embedding models used: Jina CLIP v2 (1024D, shared text/image space), BAAI/bge-large-en-v1.5
- Query processing literature: HyDE, RAG-Fusion, Multi-Query, Step-Back, Query Decomposition
- Agentic RAG frameworks (LangChain / LlamaIndex)

---

## 3. Dataset & Preprocessing

- MMDocIR subset statistics: number of PDFs, questions, answer types (Pure-text, table, image-based)
- Data split policy (train vs test — no fine-tuning on test)
- PDF extraction: unstructured library, block-level parsing
- Chunking strategies implemented:

| Strategy | Parameters | Chunk count | Avg chars |
|---|---|---|---|
| Fixed-size | 1000 chars | | |
| Sliding window | 1000 chars, 200 overlap | | |
| Hierarchical | Heading-aware | | |
| Semantic | Boundary detection | | |

- Preprocessing runtime

---

## 4. System 1 — Baseline RAG

**Architecture:**
PDF → unstructured → fixed-size chunks → Jina CLIP v2 (text) → Qdrant → Hybrid BM25 + Dense RRF → qwen3:32b

**Key design decisions:**
- Cosine distance metric
- TOP-K = 5
- RRF constant K = 60
- Fixed-size pre-processed chunks

**Prompt strategy:** Concise direct answering — "Answer using only the provided context, no padding."

**Evaluation scope:** Pure-text questions only (justified: system is text embeddings only)

| Metric | Value |
|---|---|
| P@1 | |
| P@3 | |
| P@5 | |
| R@1 | |
| R@3 | |
| R@5 | |
| Page Recall@5 | |
| Exact Match | |
| Token F1 | |
| Contains Match | |

---

## 5. System 2 — Advanced mRAG

### 5.1 Query Processing Techniques

Brief description of each technique, then results:

| Technique | P@5 | R@5 | Exact Match | Token F1 |
|---|---|---|---|---|
| Standard (baseline) | | | | |
| Multi-Query | | | | |
| RAG-Fusion (RRF) | | | | |
| HyDE | | | | |
| Step-Back | | | | |
| Query Decomposition | | | | |
| Query Rewriting | | | | |
| Query Expansion | | | | |

### 5.2 Chunking Strategy Ablation

Compare fixed-size vs sliding window vs hierarchical vs semantic on retrieval metrics. Justify the final choice.

| Chunking Strategy | P@5 | R@5 | Notes |
|---|---|---|---|
| Fixed-size | | | Baseline |
| Sliding window | | | |
| Hierarchical | | | |
| Semantic | | | |

### 5.3 Multimodal Retrieval

- Text-only retrieval: Jina CLIP v2 text mode (System 1)
- Multimodal retrieval: Jina CLIP v2 image + text shared embedding space
- Description of cross-modal matching method

| Modality | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|---|
| Text-only | | | | | | |
| Multimodal (image+text) | | | | | | |

### 5.4 Prompting Strategy Comparison

| Strategy | Description | Exact Match | Token F1 |
|---|---|---|---|
| Concise direct | "Answer only, no padding" | | |
| Chain-of-thought | Step-by-step reasoning | | |
| Role prompting | Domain expert persona | | |
| Few-shot (optional) | In-context examples | | |

### 5.5 Full Advanced System — Best Configuration

- Best combination: optimal chunking + query technique + retrieval modality + prompt
- Final comparison against System 1 baseline

| System | P@5 | R@5 | Exact Match | Token F1 |
|---|---|---|---|---|
| System 1 — Baseline | | | | |
| System 2 — Advanced (best) | | | | |

---

## 6. System 3 — Agentic mRAG

**Framework:** LangChain / LlamaIndex

**Agent architecture (minimum 3 agents):**
- Agent 1: Query Rewriting Agent — rewrites and expands the input query using LLM reasoning
- Agent 2: Retrieval Agent — decides retrieval strategy and executes search
- Agent 3: Answer Generation Agent — synthesises retrieved evidence into a final answer

Each agent involves LLM-based decision-making or control flow, not just modular function calls.

**Generation model:** qwen3-vl:8b on UiS Ollama cluster

| System | P@5 | R@5 | Exact Match | Token F1 |
|---|---|---|---|---|
| System 1 — Baseline | | | | |
| System 2 — Advanced | | | | |
| System 3 — Agentic | | | | |

---

## 7. Evaluation Framework

**Retrieval metrics** (evaluated on retrieved document sets):
- Precision@k for k = 1, 3, 5
- Recall@k for k = 1, 3, 5
- Page Recall@k
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)

**Generation metrics** (evaluated on generated answers):
- Exact Match (after normalisation: lowercasing, punctuation removal, number normalisation)
- Token F1 (SQuAD-style token overlap)
- Contains Match (ground truth appears anywhere in prediction)

**Supplementary metrics:**
- BLEU
- ROUGE-1, ROUGE-2, ROUGE-L
- Semantic similarity (sentence-transformers cosine)

**Evaluation subset:** First 20 pure-text questions from train split (justified by time constraints; discussion of limitations included)

---

## 8. Required Comparisons (per §11 of project specification)

| # | Comparison | Systems / Settings | Key Metrics |
|---|---|---|---|
| 1 | Baseline vs Advanced vs Agentic | S1 vs S2-best vs S3 | P@5, Token F1 |
| 2 | With vs without query processing | Standard vs best technique | P@5, R@5 |
| 3 | With vs without advanced chunking | Fixed-size vs best strategy | P@5, R@5 |
| 4 | Text-only vs multimodal retrieval | Text embed vs image+text embed | P@5, R@5 |
| 5 | Prompting strategy | 3+ strategies | Exact Match, Token F1 |

---

## 9. Runtime & Resource Analysis

| Phase | System 1 | System 2 | System 3 |
|---|---|---|---|
| PDF extraction + chunking | | | N/A |
| Index build time | | | |
| Avg retrieval time per query | | | |
| Avg generation time per query | | | |
| Total pipeline time per query | | | |

- Notes on GPU usage (if applicable)
- Random seed: 42 (fixed for reproducibility)
- HyDE is expected to be slowest due to N × LLM calls before retrieval; standard is fastest

---

## 10. Conclusion

- Best-performing configuration and justification
- Key findings (e.g., "RAG-Fusion improved P@5 by X% over standard; HyDE achieved best R@5")
- Limitations:
  - Evaluation on 20-question subset
  - No fine-tuning performed
  - Multimodal retrieval limited to available image collections
- Future work:
  - Agentic routing between text-only and multimodal retrieval
  - Fine-tuned embeddings on domain data
  - Larger evaluation set (full test split)

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- MMDocIR dataset paper and documentation (https://mmdocrag.github.io/MMDocIR/)
- Jina CLIP v2: Unified Multimodal Embeddings
- BAAI/bge-large-en-v1.5 embedding model
- Qdrant vector database documentation
- Gao et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
- RAG-Fusion (Rackauckas, 2023)
- Step-Back Prompting (Zheng et al., 2023)
- qwen3-vl:8b model documentation

---

## Appendix

### A — Prompt Templates

**System 1 / System 2 — Concise Direct Prompt:**

You are a concise assistant. Answer using ONLY the provided context.
Strict Instructions: Read the whole context and think before answering. NEVER add preamble, explanation, or context. ANSWER ONLY. DO NOT rephrase, explain, or add context. For yes/no questions, answer only "Yes" or "No." Do not rely only on explicit statements. If the answer can be derived through calculation, compute it before answering.

**Chain-of-Thought Prompt:**
(Insert CoT variant here)

**Role Prompting Prompt:**
(Insert role variant here)

---

### B — Configuration Summary

| Parameter | System 1 (Baseline) | System 2 (Advanced) |
|---|---|---|
| Embedding model | jinaai/jina-clip-v2 | BAAI/bge-large-en-v1.5 |
| Embedding dimension | 1024 | 1024 |
| Chunking strategy | fixed-size (1000 chars) | (best from ablation) |
| Retrieval method | Hybrid BM25 + Dense RRF | Hybrid + query technique |
| TOP-K | 5 | 5 |
| LLM model | qwen3:32b | qwen3:32b |
| LLM temperature | 0.0 | 0.0 |
| Random seed | 42 | 42 |

---

### C — Per-Question-Type Results

| Question Type | System | P@5 | R@5 | Exact Match |
|---|---|---|---|---|
| Pure-text | S1 | | | |
| Pure-text | S2 | | | |
| Table-based | S2 | | | |
| Image-based | S2 | | | |

---

### D — Reproduction Instructions

To run the baseline system:
- python src/main_baseline.py

To run the advanced system with a specific query technique:
- python src/main_advanced.py --technique hyde
- python src/main_advanced.py --technique rag_fusion
- python src/main_advanced.py --technique multi_query
- python src/main_advanced.py --technique step_back
- python src/main_advanced.py --technique query_decomposition
- python src/main_advanced.py --technique query_rewriting
- python src/main_advanced.py --technique query_expansion

To rebuild the index from scratch, set force_rebuild=True in the respective main file.
