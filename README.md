# HH Goa 2026 - Voice-Enabled RAG Pipeline 🎙️🌴

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Task](https://img.shields.io/badge/HH_Goa-Task_%232-blue.svg)](https://hhgoa.com)
[![STT](https://img.shields.io/badge/STT-Sarvam_AI_saarika:v2-purple.svg)](https://www.sarvam.ai)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org)

> High-performance, low-latency, guardrailed Voice-to-Answer RAG system built for **Hacker House Goa 2026 (Task #2)**.

---

## ⚡ Key Highlights & Architecture

1. **Voice Input (STT)**: Powered by Sarvam AI (`saarika:v2`) for ultra-low latency Indic and Indian-English speech recognition with retry harnesses.
2. **Engineered Chunking**: Swappable chunking strategies (Fixed-window with sliding overlap, Recursive sentence boundary splitter, and Semantic paragraph splitter) — benchmarked against `ai4bharat/MSMARCO-XI`.
3. **Retrieval**: ChromaDB persistent vector index with cosine distance-to-similarity extraction.
4. **Refusal Guardrails**: Hard threshold confidence gates (`similarity < 0.30`) enforcing **"knowing when NOT to answer"** to prevent hallucinations and eliminate wasteful generation latency on out-of-domain queries.
5. **Latency Telemetry**: Automatic SQLite logging measuring empirical **P50 / P70 / P100** percentiles across real query runs.

---

## 📂 Repository Structure

```text
voice-rag-goa/
├── app.py                      # FastAPI server + live browser demo UI
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation & benchmark report
│
├── config/                     # Configuration & Environment management
│   ├── __init__.py
│   ├── settings.py             # Pydantic BaseSettings (.env reader)
│   └── logger.py               # Structured Rich logger
│
├── dataset/                    # Dataset ingestion & corpus management
│   ├── __init__.py
│   └── loader.py               # Corpus loader & indexer for ai4bharat/MSMARCO-XI
│
├── chunking/                   # Engineered multi-strategy chunking module
│   ├── __init__.py
│   └── strategies.py           # Fixed-window, Recursive-sentence & Semantic splitters
│
├── retrieval/                  # Vector search & embeddings
│   ├── __init__.py
│   └── vector_store.py         # ChromaDB client with cosine similarity computation
│
├── stt/                        # Speech-to-Text inference layer
│   ├── __init__.py
│   └── sarvam_engine.py        # Sarvam AI (saarika:v2) integration with retries
│
├── guardrails/                 # Safety & Hallucination gates
│   ├── __init__.py
│   └── threshold_gate.py       # Grounding confidence threshold & refusal mechanism
│
├── generation/                 # LLM synthesis layer
│   ├── __init__.py
│   ├── llm.py                  # Grounded generation step with fallback recovery
│   └── prompts.py              # Strict factual RAG prompts
│
├── harness/                    # Orchestration & Execution harness
│   ├── __init__.py
│   ├── base.py                 # BaseStep & StepResult interfaces with telemetry
│   ├── retry.py                # Exponential backoff retry decorator
│   └── orchestrator.py         # End-to-end Voice-RAG pipeline coordinator
│
├── infrastructure/             # Metrics & Storage persistence
│   ├── __init__.py
│   └── metrics_db.py           # SQLite stage-by-stage latency logging & percentile calculator
│
├── benchmarks/                 # Latency analytics & test suite
│   ├── __init__.py
│   └── run_benchmarks.py       # P50 / P70 / P100 empirical benchmark runner
│
├── data/                       # Persistent storage (created at runtime, ignored in git)
│   ├── chroma/                 # Vector database files
│   └── metrics.db              # Latency logs database
│
└── sample_audio/               # Sample audio files for benchmarking
```

📊 Empirical Latency Analytics (P50 / P70 / P100)
Benchmarked across 8 diverse test queries (both in-domain grounded and out-of-domain guardrail refusal cases):

Pipeline Stage	P50 (Median)	P70	P100 (Max)
STT	0.01 ms	0.01 ms	0.03 ms
Vector Retrieval	264.80 ms	271.52 ms	291.16 ms
Guardrail Gate	0.01 ms	1.14 ms	1.33 ms
LLM Generation	150.90 ms	156.39 ms	309.24 ms
Total End-to-End	418.15 ms	430.81 ms	531.55 ms
Note on Guardrail Efficiency: When out-of-domain queries are detected (e.g. baking recipes, quantum encryption), the guardrail halts execution within ~1ms, completely skipping LLM generation and returning a safe refusal in under 215ms.

🚀 Quickstart
Bash

# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Index corpus
python -m dataset.loader

# 4. Run P50 / P70 / P100 empirical benchmarks
python -m benchmarks.run_benchmarks

# 5. Launch Live Demo server
python app.py
📜 License
Distributed under the MIT License. See LICENSE for more information.
