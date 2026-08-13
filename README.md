# HH Goa 2026 - Voice-Enabled RAG Pipeline 🎙️🌴

> High-performance, low-latency, guardrailed Voice-to-Answer system for **Hacker House Goa 2026 (Task #2)**.

## Architecture Highlights
- **Harness Core (`harness/base.py`)**: Typed execution step units with microsecond telemetry and retry decorators.
- **Multi-Strategy Chunking (`chunking/strategies.py`)**: Fixed-window, Recursive sentence-aware, and Semantic paragraph splitters.
- **Vector Retrieval (`retrieval/vector_store.py`)**: ChromaDB persistent vector index with cosine similarity scores.
- **Guardrails (`guardrails/threshold_gate.py`)**: Hard threshold gate (similarity < 0.65) that knows when **NOT** to answer to prevent hallucination.
- **Latency Analytics (`benchmarks/run_benchmarks.py`)**: Automatic P50 / P70 / P100 empirical measurement.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest corpus
python -m dataset.loader

# 3. Run P50/P70/P100 Benchmark Suite
python -m benchmarks.run_benchmarks

# 4. Launch Live Demo & API Server
python app.py
