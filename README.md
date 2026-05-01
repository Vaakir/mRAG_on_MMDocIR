# DAT560 Project — Multimodal RAG on MMDocIR

A three-system Retrieval-Augmented Generation pipeline evaluated on the [MMDocIR](https://github.com/MMDocIR/MMDocIR) benchmark (750 queries over 131 documents). Built for DAT 560 at the University of Stavanger.

---

## Table of Contents

- [Systems Overview](#systems-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data](#data)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproducibility](#reproducibility)

---

## Systems Overview

| System | Type | Description |
|--------|------|-------------|
| **System 1 — Baseline RAG** | Text-only | Jina CLIP v2 embeddings, Qdrant vector store, hybrid BM25 + dense retrieval, Ollama generation |
| **System 2 — Advanced mRAG** | Multimodal | Adds page image, figure, and evidence crop retrieval; 8 query techniques; 5 chunking strategies; 4 prompting strategies |
| **System 3 — Agentic mRAG** | Agent-based | LangChain & LangGraph StateGraph with query rewriter, grader, and generator agents; adaptive retry on low-confidence retrievals |

### Architecture

```
PDFs + Page Images + Figures
         │
         ▼  (one-time preprocessing)
  pdf_loader → pdf_chunker → embedder → Qdrant
                                          │
         ┌────────────────────────────────┘
         ▼
    User Query
         │
         ▼
  Query Technique  (standard / multi_query / rag_fusion / hyde / step_back / ...)
         │
         ▼
  Retrieved Chunks ± Images
         │
         ▼
  Prompt Strategy  (standard / cot / few_shot / role )
         │
         ▼
  LLM via Ollama  (qwen3-vl:8b-instruct, school-provided)
         │
         ▼
     Answer + Metrics
```

System 3 replaces the fixed pipeline with a LangGraph agent loop:

```
Query → [Query_rewriter] → [Grader] ──(low confidence)──→ retry
                                │
                          (sufficient confidence OR retries exhausted)
                                ▼
                         [Generator] → Answer
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── info/
│   ├── hours.csv
│   ├── mRAG.md               # Task specification
│   └── summary.ipynb
└── src/
    ├── main.py               # Entry point: System 1 & 2
    ├── main_agentic.py       # Entry point: System 3
    ├── .env.example
    ├── docker-compose.yml    # Optional: Qdrant via Docker
    ├── config/
    │   └── config.py         # BaselineConfig, AdvancedConfig, AgenticConfig
    ├── data/
    │   ├── train/
    │   ├── test/
    │   └── preprocessed/     # Cached chunk files (preprocessed-generated files)
    ├── preprocessing/
    │   ├── pdf_loader.py     # Docling-based PDF extraction
    |   ├── build_multimodal_indexes.py
    │   ├── pdf_chunker.py    # 5 chunking strategies
    │   ├── image_processor.py
    │   └── extract_figures.py
    ├── indexing/
    │   ├── embedder.py       # Jina CLIP v2 (text + image, 1024D)
    │   ├── vector_database.py
    │   └── hybrid_retriever.py
    ├── query_techniques/     # standard, multi_query, rag_fusion, hyde, step_back, ...
    ├── retrieval_techniques/
    │   └── multimodal.py     # Image-aware retrieval routing
    ├── generation/
    │   ├── generator.py      # BaselineGenerator + VisionGenerator
    │   ├── prompts/          # standard, cot, few_shot, role
    │   └── answer_validator.py
    ├── pipelines/
    │   ├── base_pipeline.py
    │   ├── baseline_pipeline.py
    │   ├── advanced_pipeline.py
    │   ├── agentic_pipeline.py
    │   └── preprocessing_pipeline.py
    ├── agentic/
    │   ├── llm.py
    │   ├── graph/            
    │   │   ├── builder.py    # Graph builder
    │   │   ├── nodes.py      # Node definitions
    │   │   └── state.py      # State schema
    │   └── tools/
    │       ├── output_parser.py
    ├── evaluation/
    │   ├── retrieval_metrics.py
    │   └── generation_metrics.py
    ├── utils/
    │   └── timer.py
    ├── notebooks/            # Exploration and debugging notebooks
    └── results/              # CSV output from experiments
```

---

## Setup

### Prerequisites

- Python 3.11
- Ollama access provided by the university (no local Ollama install required)
- **Conda (optional)** — Makes environment setup easier, but not required. If you don't have it, use the fallback option below

---

### Option A: Using Conda (Recommended)

Conda creates an isolated Python environment so dependencies don't conflict with your system or other projects.

```bash
# Create and activate conda environment
conda create --name dat560project python=3.11
conda activate dat560project

# Install dependencies
pip install -r requirements.txt
```

**To deactivate when done:**
```bash
conda deactivate
```

---

<details>
<summary><b>Option B: Using Python's Built-in venv (No Conda Required)</b></summary>

If you don't have Conda installed, use Python's built-in `venv` module instead.

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv dat560project
```

**Activate the environment** (choose based on your OS):

**macOS / Linux:**
```bash
source dat560project/bin/activate
```

**Windows:**
```bash
dat560project\Scripts\activate
```

**Then install dependencies:**

```bash
pip install -r requirements.txt
```

**To deactivate when done:**
```bash
deactivate
```

</details>

---

### GPU Support (Optional, Windows/Linux Only)

Only follow this step if you have an NVIDIA GPU and want faster computation. MacOS users skip this.

```bash
# Uninstall default CPU-only PyTorch
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment variables

Copy the example file and fill in your credentials:

```bash
cp src/.env.example src/.env
```

```env
OLLAMA_API_KEY=your_key_here      # provided by school
```

---

## Data

Download the MMDocIR subset and place it under `src/data/`:

```
src/data/
├── train/
│   ├── pdfs_train/
│   ├── page_images_train/
│   ├── images_train/
│   └── train.jsonl
└── test/
    ├── pdfs_test/
    ├── page_images_test/
    ├── images_test/
    └── test.jsonl
```

---

## Running the Pipeline

### Step 1 — Preprocessing (run once)

Extracts text and images from PDFs, applies all chunking strategies, and builds Qdrant indexes.

```bash
cd src

# Full run: PDF extraction + chunking + build all multimodal indexes
python -m pipelines.preprocessing_pipeline
```

**What gets created:**

By default, the script rebuilds all Qdrant collections (one per chunking strategy) but **skips PDF extraction** (assumes `all_documents.json` already exists). 

To control which stages run, edit [src/pipelines/preprocessing_pipeline.py](src/pipelines/preprocessing_pipeline.py) lines 79–80:

```python
result = preprocessing_pipeline(
    force_chunking=False,   # Set to True to re-extract PDFs + apply chunking
    force_indexing=True,    # Set to True to rebuild all index collections
)
```

This creates 5 Qdrant collections in your local database:
- `advanced_fixed_size`
- `advanced_sliding_window`
- `advanced_semantic`
- `advanced_hierarchical`
- `advanced_enhanced_hierarchical`

Each contains the same documents but chunked differently. You select which one to use when running System 1 or System 2 (see [Configuration](#configuration) below).

---

### Running preprocessing on Google Colab

Upload the cloned project to Google Drive, then mount and run:

<details>
<summary>Click to expand Colab setup script</summary>

```python
from google.colab import drive

try:
    drive.flush_and_unmount()
    print("Drive unmounted successfully (or was not mounted).")
except ValueError:
    print("Drive was not mounted or could not be unmounted, proceeding with mount attempt.")

print("Attempting to mount Google Drive...")
drive.mount('/content/drive')

import os
project_path = '/content/drive/My Drive/DAT560project'

if os.path.exists(project_path) and os.path.isdir(project_path):
    print(f"Contents of {project_path}:")
    print(os.listdir(project_path))

    # Uncomment to install requirements:
    # requirements_path = os.path.join(project_path, 'requirements.txt')
    # if os.path.exists(requirements_path):
    #     get_ipython().system(f'pip install -r "{requirements_path}"')

    script_path = os.path.join(project_path, 'src', 'pipelines', 'preprocessing_pipeline.py')

    if os.path.exists(script_path):
        get_ipython().system(f'python "{script_path}"')
    else:
        print(f"The script '{script_path}' does not exist. Please check the path.")
else:
    print(f"The folder '{project_path}' does not exist or is not a directory after mounting.")
    print("Please verify the exact path of your project folder and ensure it's shared correctly.")
```

</details>

---

### Step 2 — Run Systems 1 & 2

#### System 1: Baseline RAG (Text-only)

To run **System 1 only**, uncomment the code block below in `main.py` (lines 320–329) and **comment out everything else** in the `else:` statement:

```python
# TO RUN BASELINE (System 1):
config = BaselineConfig()
pure_text_data = load_train_data(config.TEST_JSONL)
run_single_experiment(
    experiment_name="Baseline RAG",
    config=config,
    pipeline_class=BaselineRAGPipeline,
    test_data=pure_text_data[:config.EVAL_SUBSET_SIZE],
    force_rebuild=False,
    run_single_query=True
)
```

Then run:

```bash
cd src
python main.py
```

This uses the baseline configuration from `src/config/config.py` (fixed_size chunking, standard query technique, standard prompting).

---

#### System 2: Advanced mRAG (Multimodal)

**Option A: Run with Configuration**

Uncomment this block in `main.py` (lines 339–353) and comment out everything else:

```python
# TO RUN:
config = AdvancedConfig(
    CHUNKING_STRATEGY="enhanced_hierarchical",  # or "semantic"
    QUERY_TECHNIQUE="rag_fusion",               # or "hyde", "multi_query", etc.
    PROMPTING_STRATEGY="cot",                   # Chain of Thought
    USE_MULTIMODAL=True                         # If its to use images or not.
)
pure_text_data = load_train_data(config.TEST_JSONL)
run_single_experiment(
    experiment_name="Advanced RAG (Best Config)",
    config=config,
    pipeline_class=AdvancedRAGPipeline,
    test_data=pure_text_data,
    force_rebuild=False,
    run_single_query=True
)
```

Then run:

```bash
cd src
python main.py
```

This runs **one experiment** with your chosen parameters. You can modify `CHUNKING_STRATEGY`, `QUERY_TECHNIQUE`, and `PROMPTING_STRATEGY` in this block to test different configurations.

---

**Option B: Run ALL Ablation Experiments**

Uncomment this block in `main.py` (line 333) and comment out everything else:

```python
# TO RUN ALL ABLATION TESTS:
run_experiments(eval_subset_size=1000)
```

Then run:

```bash
cd src
python main.py
```

This systematically runs:
- 1 Baseline RAG
- 4 Chunking strategy ablations (sliding_window, semantic, hierarchical, enhanced_hierarchical)
- 7 Query technique ablations (multi_query, rag_fusion, step_back, hyde, query_decomposition, query_rewriting, query_expansion)
- 3 Prompting strategy ablations (few_shot, role, cot)
- 1 Multimodal variant

---

**Option C: Run Incremental Optimization (Greedy Search)**

Uncomment this block in `main.py` (line 336) and comment out everything else:

```python
# TO RUN INCREMENTAL ADDITION (Step-by-step optimization):
run_incremental_addition(eval_subset_size=1000, target_metric='token_f1')
```

Then run:

```bash
cd src
python main.py
```

This uses **greedy optimization**: sequentially tests each component (chunking, query technique, prompting strategy, multimodal), carrying forward only the best-performing option at each step. Much faster than all ablations, but less comprehensive.

---

#### Run Single Experiments from CLI

**Instead of editing main.py**, you can run single experiments directly from the terminal using the `--run-experiments` flag:

```bash
# EXAMPLES:
python main.py --run-experiments --technique multi_query --prompting-strategy cot

python main.py --run-experiments --technique rag_fusion --eval-subset 30

python main.py --run-incremental --incremental-metric exact_match

python main.py --run-experiments --technique hyde --force-rebuild
```

**Important:** When using CLI flags, you must include `--run-experiments` or `--run-incremental` for them to take effect. Without these flags, the hardcoded default config runs instead.

---

#### CLI Flags Reference

| Flag | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| `--run-experiments` | Flag | — | Run a single experiment with specified technique and strategy | `--run-experiments --technique multi_query` |
| `--run-incremental` | Flag | — | Run greedy optimization selecting best at each step | `--run-incremental` |
| `--technique` | String | standard | Query technique (only used with --run-experiments) | `--technique rag_fusion` |
| `--prompting-strategy` | String | standard | Prompting strategy (only used with --run-experiments) | `--prompting-strategy cot` |
| `--eval-subset` | Integer | 20 | Number of test queries to evaluate | `--eval-subset 50` |
| `--incremental-metric` | String | token_f1 | Target metric to optimize (only used with `--run-incremental`) | `--incremental-metric exact_match` |
| `--force-rebuild` | Flag | — | Force rebuild of Qdrant index | `--force-rebuild` |

---

**Results** are saved to:
- Single experiments: `src/results/pipeline_results.csv`
- All ablations: `src/results/results_experiments.csv`
- Incremental optimization: `src/results/pipeline_results.csv` (timestamped rows)

---

#### Text-Only Mode (Disable Multimodal)

**For Option A (code block):** Change `USE_MULTIMODAL=False` directly in the code block:

```python
config = AdvancedConfig(
    CHUNKING_STRATEGY="enhanced_hierarchical",
    QUERY_TECHNIQUE="rag_fusion",
    PROMPTING_STRATEGY="cot",
    USE_MULTIMODAL=False  # Change to False here
)
```

**For Options B, C, and CLI modes:** Edit [src/config/config.py](src/config/config.py) and change `USE_MULTIMODAL` in the `AdvancedConfig` class:

```python
USE_MULTIMODAL: bool = False
```

Then run any of these commands (EXAMPLES):

```bash
python main.py --run-experiments --technique rag_fusion

python main.py --run-incremental --incremental-metric token_f1

python main.py  # (if you uncommented Option B or C)
```

This completely disables multimodal retrieval—no images are retrieved or processed, only text.

---

### Step 3 — Run System 3 (Agentic)

```bash
cd src

# Testing with a single query
python main_agentic.py --test-query "How many students of NTU would recommend studying at NTU?"

# Running the full evaluation (on test set)
python main_agentic.py --eval --eval-size 150 --output results_agentic.json
```

The agentic system prints each agent's decision (query technique chosen, grader confidence, prompting strategy), per query.
You can also specify a `JSON` output file to store information and output, if further inspection is necessary. 

*Note: Running the above-mentioned System 3 commands assumes that the index is already built.*

---

## Configuration

All settings live in `src/config/config.py`. Three config classes are available: `BaselineConfig` (System 1), `AdvancedConfig` (System 2), `AgenticConfig` (System 3).

### Selecting a Chunking Strategy & Collection

`CHUNKING_STRATEGY` and `VECTOR_DB_COLLECTION` are **independent settings** that must match:

| CHUNKING_STRATEGY | Should use Collection | Notes |
|-------------------|---------------------|-------|
| `fixed_size` | `advanced_fixed_size` | Fixed-size chunks |
| `sliding_window` | `advanced_sliding_window` | Sliding window chunks |
| `semantic` | `advanced_semantic` | Semantic boundary chunks |
| `hierarchical` | `advanced_hierarchical` | Hierarchical chunks |
| `enhanced_hierarchical` | `advanced_enhanced_hierarchical` | Enhanced hierarchical chunks |

**When using `--run-experiments` from CLI:** The collection is set automatically based on `CHUNKING_STRATEGY`:

```bash
python main.py --run-experiments --technique rag_fusion --chunking-strategy semantic
# Automatically sets: VECTOR_DB_COLLECTION = "advanced_semantic"
```

**When running System 1 or System 2 directly** (without `--run-experiments`): You must manually set both in [src/config/config.py](src/config/config.py) to match:

```python
# In AdvancedConfig (around line 135):
CHUNKING_STRATEGY: str = "semantic"
VECTOR_DB_COLLECTION: str = "advanced_semantic"  # Must match!
```

**Important:** The collection must already exist from preprocessing (see [Step 1](#step-1--preprocessing-run-once) above). Alternatively, you can rebuild it with the `--force-rebuild` flag:

```bash
python main.py --run-experiments --technique rag_fusion --force-rebuild
```

---

### Configuration Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `CHUNKING_STRATEGY` | `fixed_size` | `fixed_size`, `sliding_window`, `semantic`, `hierarchical`, `enhanced_hierarchical` |
| `QUERY_TECHNIQUE` | `standard` | `standard`, `multi_query`, `rag_fusion`, `step_back`, `hyde`, `query_rewriting`, `query_expansion`, `query_decomposition` |
| `PROMPTING_STRATEGY` | `standard` | `standard`, `cot`, `few_shot`, `role`, `ensemble` |
| `USE_MULTIMODAL` | `True` | Enable multimodal retrieval (page images, figures, evidence crops) |
| `TOP_K` | `5` | Number of retrieved chunks per query |
| `EVAL_SUBSET_SIZE` | `20` | Number of test questions to evaluate |
| `VECTOR_DB_MODE` | `local` | `local` (disk), `memory` (RAM), `docker` (remote Qdrant) |
| `LLM_MODEL` | `qwen3-vl:8b-instruct` | Any model available via Ollama |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature (reduce hallucination) |
| `AGENT_MAX_RETRIES`| `1` | Specify number of retry-attempts available |
| `GRADER_CONFIDENCE_THRESHOLD`| `0.51` | Specify retry confidence threshold |
---

## Evaluation Metrics

### Retrieval

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of retrieved docs in top-k that are relevant |
| Recall@k | Fraction of all relevant docs that appear in top-k |
| Page Recall@k | Binary hit: 1.0 if any retrieved chunk in top-k matches relevant PDF and overlaps relevant pages, else 0.0 |
| MAP | Mean Average Precision across all queries; average precision at each position where a relevant doc is found |
| MRR | Mean Reciprocal Rank of the first relevant result (1 / rank of first hit) |
| NDCG@k | Normalized Discounted Cumulative Gain; ranking quality with position weighting (DCG@k / IDCG@k) |

### Generation

| Metric | Description |
|--------|-------------|
| Exact Match | Normalized string equality with ground truth (after lowercasing, punctuation removal, etc.) |
| Contains Match | Binary: 1.0 if ground truth value appears anywhere in the prediction, else 0.0 |
| Token F1 | SQuAD-style token-level F1 score; measures token overlap frequency between prediction and answer |
| BLEU | Bilingual Evaluation Understudy; n-gram overlap score (BLEU-4: up to 4-grams) with brevity penalty |
| ROUGE-1 | Recall-Oriented Understudy for Gisting Evaluation; unigram (1-word) overlap F-measure |
| ROUGE-2 | ROUGE for bigrams (2-word sequences) F-measure |
| ROUGE-L | ROUGE for longest common subsequence F-measure |
| Semantic (Textual) Similarity | Embedding-based cosine similarity between prediction and answer embeddings|

---

## Reproducibility

| Mechanism | Detail |
|-----------|--------|
| Prompt templates | Versioned in `src/generation/prompts/` |
| Preprocessing artifacts | Chunk files cached in `src/data/preprocessed/` and reused |
| Index reuse | Qdrant collections reused unless `--force-rebuild` is passed |
| Deterministic generation | `LLM_TEMPERATURE = 0.0`, `LLM_TOP_P = 0.1` |
| Timing logs | Preprocessing and experiment runtimes saved to `src/results/` |

---

## Remove Environment

```bash
conda remove --name dat560project --all
```