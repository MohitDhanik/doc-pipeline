# Document Ingestion & Retrieval Data Pipeline

An end-to-end ETL pipeline for unstructured data — parse, chunk, embed, quality-check, store in a vector database, and evaluate retrieval quality.

## What it does

Takes raw documents (PDF, TXT, Markdown, CSV) and makes them semantically searchable:

```
Raw Documents → Parse → Chunk → Embed → QA Checks → FAISS Store → Semantic Retrieval → Evaluation
```

- **Document parsing** — handles PDF, TXT, MD, CSV formats
- **Recursive chunking** — 512-char chunks with 64-char overlap
- **Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (local, no API key needed)
- **Quality checks** — null content, minimum length, exact deduplication
- **Vector store** — FAISS `IndexFlatL2`; ChromaDB as alternative
- **Evaluation** — precision@k and recall@k across k = 1, 3, 5, 10

## Project structure

```
src/
  ingestion/      document_parser.py, chunker.py
  transformation/ schema.py, embedder.py, quality_checks.py
  store/          faiss_store.py, chroma_store.py
  retrieval/      retriever.py, evaluator.py
  pipeline.py     end-to-end orchestrator
data/raw/         6 sample documents
tests/            28 pytest tests
app.py            Streamlit dashboard
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline

```bash
# Pre-build the FAISS index on sample documents
PYTHONPATH=. python src/pipeline.py

# Launch the dashboard
PYTHONPATH=. streamlit run app.py
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Stack

Python · FAISS · ChromaDB · sentence-transformers · pdfplumber · Streamlit · pytest
