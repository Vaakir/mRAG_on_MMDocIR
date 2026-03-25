# DAT560 Project

**Deadline:** Week 17 Monday (20.04.2026)

---

## Filestructure Proposal

```text
src/
│
├── data/
│   ├── loaders.py
│   ├── pdf_parser.py
│   ├── image_extractor.py
│   └── dataset.py
│
├── preprocessing/
│   ├── chunking.py
│   ├── cleaning.py
│   └── multimodal_alignment.py
│
├── indexing/
│   ├── embedder.py
│   ├── vector_store.py
│   ├── multimodal_index.py
│   └── build_index.py
│
├── retrieval/
│   ├── retriever.py
│   ├── multimodal_retriever.py
│   ├── query_processing.py
│   └── reranker.py
│
├── generation/
│   ├── generator.py
│   ├── prompts/
│   │   ├── baseline.txt
│   │   ├── cot.txt
│   │   ├── fewshot.txt
│   │   └── role.txt
│   └── llm_wrapper.py
│
├── agentic/
│   ├── agents.py
│   ├── planner.py
│   ├── tools.py
│   └── pipeline.py
│
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── generation_metrics.py
│   └── evaluator.py
│
├── pipelines/
│   ├── baseline_pipeline.py
│   ├── advanced_pipeline.py
│   └── agentic_pipeline.py
│
└── utils/
    ├── logging.py
    ├── caching.py
    └── seed.py