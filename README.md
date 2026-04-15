# DAT560 Project

**Deadline:** Week 17 Monday (20.04.2026)

---

## How to run
- Run all DAT560project/src> `python run_experiments.py`
- Run individial DAT560project/src> `python main_baseline.py`


## Project Structure

```text
.
├── README.md
├── requirements.txt
├── requirements2.txt
├── info/
│   ├── hours.csv
│   ├── mRAG.md
│   └── summary.ipynb
├── local_qdrant/
│   └── ...
└── src/
    ├── docker-compose.yml
    ├── main_advanced.py
    ├── main_baseline.py
    ├── config/
    │   ├── __init__.py
    │   └── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── chunk_loader.py
    │   └── data_loader.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── generation_metrics.py
    │   └── retrieval_metrics.py
    ├── generation/
    │   ├── __init__.py
    │   ├── generator.py
    │   └── prompts/
    │       ├── __init__.py
    │       ├── base.py
    │       ├── cot.py
    │       ├── ensemble.py
    │       ├── few_shot.py
    │       ├── role.py
    │       └── standard.py
    ├── indexing/
    │   ├── __init__.py
    │   ├── embedder.py
    │   ├── embedder_bge_large.py
    │   ├── embedder_clip.py
    │   ├── embedder_jina.py
    │   ├── embedder_old.py
    │   ├── hybrid_retriever.py
    │   └── vector_database.py
    ├── pipelines/
    │   ├── __init__.py
    │   ├── advanced_pipeline.py
    │   ├── base_pipeline.py
    │   ├── baseline_pipeline.py
    │   └── preprocessing_pipeline.py
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── pdf_chunker.py
    │   └── pdf_loader.py
    └── query_techniques/
        ├── __init__.py
        ├── base.py
        ├── hyde.py
        ├── multi_query.py
        ├── query_decomposition.py
        ├── query_expansion.py
        ├── query_rewriting.py
        ├── rag_fusion.py
        ├── standard.py
        └── step_back.py
```