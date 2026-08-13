# HH Goa 2026 - Voice-Enabled RAG Pipeline 🎙️🌴

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Task](https://img.shields.io/badge/HH_Goa-Task_%232-blue.svg)](https://hhgoa.com)
[![STT](https://img.shields.io/badge/STT-ElevenLabs_%26_Sarvam_AI-purple.svg)](https://elevenlabs.io)
[![Dataset](https://img.shields.io/badge/Dataset-ai4bharat%2FMSMARCO--XI-orange.svg)](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
[![LLM](https://img.shields.io/badge/LLM-Groq_llama--3.1--8b-blue.svg)](https://groq.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org)

> High-performance, low-latency, guardrailed Voice-to-Answer RAG system built for **Hacker House Goa 2026 (Task #2)**.

---

## ⚡ Key Highlights & Architecture

1. **Voice Input (STT)**: Multi-engine support using ElevenLabs (`scribe_v2`) first, with Sarvam AI (`saaras:v3`) as a fallback. Both providers are wrapped by the pipeline retry harness.
2. **Engineered Chunking (Task Requirement #2)**: Swappable chunking strategies (Fixed-window with sliding overlap, Recursive sentence boundary splitter, and Semantic paragraph splitter) empirically evaluated on `ai4bharat/MSMARCO-XI`.
3. **Retrieval**: ChromaDB persistent vector index with cosine distance-to-similarity extraction over 344 authentic `ai4bharat/MSMARCO-XI` passages (97,269 characters).
4. **Refusal Guardrails (Task Requirement #6)**: Configurable confidence gates enforce **"knowing when NOT to answer"** to prevent hallucinations and eliminate wasteful generation latency on out-of-domain queries.
5. **Execution Harness & Resilience (Task Requirement #5)**: Typed `BaseStep`/`StepResult` unit orchestration with exponential backoff and jitter to survive API rate limits under heavy burst load.
6. **Empirical Latency Telemetry (Task Requirement #4)**: Automatic SQLite logging measuring empirical **P50 / P70 / P100** percentiles across real query runs.

---

## 🔬 Full Corpus Chunking Strategies Evaluation (Task Requirement #2)

Empirically evaluated on the complete **`ai4bharat/MSMARCO-XI`** dataset across 344 passages (97,269 characters):

| Strategy Name | Total Chunks | Avg Chunk Length | Boundary / Overlap Type | Execution Latency |
| :--- | :---: | :---: | :--- | :---: |
| **`fixed_window`** | 406 | 299.1 chars | Sliding Window (60 char overlap) | `0.36 ms` |
| **`recursive_sentence`** | 306 | 315.8 chars | Syntactic Sentence Boundary | `2.88 ms` |
| **`semantic_paragraph`** | 344 | 280.8 chars | Paragraph / Structural Split | `0.50 ms` |

---

## 📊 Empirical Latency Analytics (P50 / P70 / P100) (Task Requirement #4)

Benchmarked across **35 diverse queries** (30 authentic queries directly from `ai4bharat/MSMARCO-XI` + 5 out-of-domain guardrail refusal cases):

| Pipeline Stage | P50 (Median) | P70 | P100 (Max) |
| :--- | :--- | :--- | :--- |
| **STT (ElevenLabs / Sarvam)** | `0.01 ms` | `0.01 ms` | `0.02 ms` |
| **Vector Retrieval** | `207.98 ms` | `223.18 ms` | `373.25 ms` |
| **Guardrail Gate** | `0.01 ms` | `0.01 ms` | `1.17 ms` |
| **LLM Generation (Groq LLaMA-3.1)** | `422.53 ms` | `3304.30 ms` | `5570.66 ms` |
| **Total End-to-End** | **`676.05 ms`** | **`3517.45 ms`** | **`5813.15 ms`** |

> **Note on Guardrail Efficiency:** When out-of-domain queries are detected (e.g. baking recipes, quantum encryption, World Cup queries), the guardrail halts execution in **~0.01ms**, completely skipping LLM generation and returning a safe refusal in under **188ms**.

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
│   ├── strategies.py           # Fixed-window, Recursive-sentence & Semantic splitters
│   └── benchmark.py            # Comparative chunking evaluator
│
├── retrieval/                  # Vector search & embeddings
│   ├── __init__.py
│   └── vector_store.py         # ChromaDB client with cosine similarity computation
│
├── stt/                        # Speech-to-Text inference layer
│   ├── __init__.py
│   └── engine.py               # Unified ElevenLabs and Sarvam AI STT integration
│
├── guardrails/                 # Safety & Hallucination gates
│   ├── __init__.py
│   └── threshold_gate.py       # Grounding confidence threshold & refusal mechanism
│
├── generation/                 # LLM synthesis layer
│   ├── __init__.py
│   ├── llm.py                  # Grounded generation step with Groq LLaMA-3.1 & fallback
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
│   ├── test_queries.json       # 30 Authentic MSMARCO-XI extracted queries
│   └── run_benchmarks.py       # 35-Query P50 / P70 / P100 empirical benchmark runner
│
└── data/                       # Persistent storage (ignored in git)
    ├── chroma/                 # Vector database files (344 indexed passages)
    └── metrics.db              # Latency logs database
```

## 🚀 Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Add GROQ_API_KEY and at least one STT key:
# ELEVENLABS_API_KEY=... and/or SARVAM_API_KEY=...

# 4. Ingest MSMARCO-XI corpus
python -m dataset.loader --samples 30 --strategy recursive_sentence

# 5. Run chunking evaluation and the 35-query latency benchmark
python -m chunking.benchmark
python -m benchmarks.run_benchmarks

# 6. Launch the live demo server
python app.py
```

The demo is served at `http://127.0.0.1:8000`. It accepts live microphone input, uploaded audio, or a text-only query. `GET /api/metrics` returns the latency percentiles recorded in SQLite.

## 🔐 Configuration

Copy `.env.example` to `.env` and set the values you need:

| Variable | Purpose |
| :--- | :--- |
| `ELEVENLABS_API_KEY` | Primary STT provider, using ElevenLabs Scribe v2. |
| `SARVAM_API_KEY` | STT fallback provider, using Sarvam `saaras:v3`. |
| `GROQ_API_KEY` | Enables Groq-backed grounded generation; otherwise the app uses its extractive fallback. |
| `HF_TOKEN` | Optional Hugging Face token for dataset access. |
| `SIMILARITY_THRESHOLD` | Retrieval confidence required before generation (default: `0.25`). |

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
