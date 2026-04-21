# DAT560 Project — Multimodal RAG on MMDocIR

**Deadline:** Week 17 Monday (20.04.2026)

A three-system Retrieval-Augmented Generation pipeline evaluated on the MMDocIR benchmark (750 queries over 131 documents).

---

## Systems Overview

| System | Description |
|--------|-------------|
| **System 1 — Baseline RAG** | Text-only pipeline: Jina CLIP v2 embeddings, Qdrant vector store, hybrid BM25+RRF retrieval, Ollama generation |
| **System 2 — Advanced mRAG** | Adds multimodal retrieval (page images, figures, evidence crops), 8 query techniques, 5 chunking strategies, 5 prompting strategies |
| **System 3 — Agentic mRAG** | LangGraph StateGraph with query rewriter, grader, and generator agents |

---

## Setup

```bash
# Create environment
conda create --name dat560project python=3.11
conda activate dat560project

# Install dependencies
pip install -r requirements.txt

# GPU support (non-Mac only)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Models (via Ollama)

```bash
ollama pull qwen3-vl:8b-instruct   # used by all three systems
```

### Environment variables

Create a `.env` file in `src/`:

```
OLLAMA_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token
```

---

## Data

Download the MMDocIR subset and place it under `src/data/`:

```
src/data/
├── train/
│   ├── pdf_train/
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

```bash
cd src

# Full run: PDF extraction + chunking + build all 5 multimodal indexes
python -m pipelines.preprocessing_pipeline #configurable in code
```

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

### Step 2 — Run experiments

```bash
cd src

# Run all configured experiments
python run_experiments.py

```

Results are saved to `src/results/pipeline_results.csv`.

---

## Configuration

All settings are in `src/config/config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNKING_STRATEGY` | `fixed_size` | `fixed_size` \| `sliding_window` \| `semantic` \| `hierarchical` \| `enhanced_hierarchical` |
| `QUERY_TECHNIQUE` | `standard` | `standard` \| `hyde` \| `multi_query` \| `rag_fusion` \| `step_back` \| `query_rewriting` \| `query_expansion` \| `query_decomposition` |
| `PROMPTING_STRATEGY` | `standard` | `standard` \| `cot` \| `few_shot` \| `role` \| `ensemble` |
| `USE_MULTIMODAL` | `True` | Enable multimodal retrieval (page images, figures, evidence) |
| `TOP_K` | `5` | Number of retrieved chunks |
| `EVAL_SUBSET_SIZE` | `20` | Number of test questions to evaluate |

---

## Reproducibility

- **Random seed**: fixed at `RANDOM_SEED = 42` in config
- **Prompt templates**: versioned in `src/generation/prompts/`
- **Preprocessing artifacts**: chunk files saved to `src/data/preprocessed/` and reused on subsequent runs
- **Index reuse**: Qdrant collections are reused unless `--force` is passed
- **Deterministic generation**: `LLM_TEMPERATURE = 0.0`, `LLM_TOP_P = 0.1`
- **Timing**: preprocessing and experiment runtimes logged to `src/results/`

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── info/
│   ├── hours.csv
│   └── mRAG.md
│   └── summary.ipynb
└── src/
    ├── main.py
    ├── config/
    ├── data/
    ├── evaluation/
    ├── generation/
    │   ├── answer_validator.py
    │   ├── generator.py
    │   └── prompts/
    ├── indexing/
    │   ├── embedder.py
    │   ├── embedder_bge_large.py
    │   ├── embedder_clip.py
    │   ├── hybrid_retriever.py
    │   └── vector_database.py
    ├── notebooks/
    ├── pipelines/
    │   ├── advanced_pipeline.py
    │   ├── agentic_pipeline.py
    │   ├── base_pipeline.py
    │   ├── baseline_pipeline.py
    │   └── preprocessing_pipeline.py
    ├── preprocessing/
    │   ├── build_multimodal_indexes.py
    │   ├── extract_figures.py
    │   ├── image_processor.py
    │   ├── pdf_chunker.py
    │   └── pdf_loader.py
    ├── query_techniques/
    ├── retrieval_techniques/
    │   └── multimodal.py
    └── utils/
        └── timer.py
```

---

## If necessary (delete environment)

```bash
conda remove --name dat560project --all
```
