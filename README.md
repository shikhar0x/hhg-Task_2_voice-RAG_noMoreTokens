# HH Goa 2026 - Voice-Enabled RAG Pipeline 🎙️🌴

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Task](https://img.shields.io/badge/HH_Goa-Task_%232-blue.svg)](https://hhgoa.com)
[![STT](https://img.shields.io/badge/STT-ElevenLabs_Scribe_v2-purple.svg)](https://elevenlabs.io)
[![Dataset](https://img.shields.io/badge/Dataset-ai4bharat%2FMSMARCO--XI-orange.svg)](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
[![LLM](https://img.shields.io/badge/LLM-Groq_Meta_LLaMA--3.1--8B-blue.svg)](https://groq.com)
[![Retrieval Latency](https://img.shields.io/badge/Local_Retrieval-P50_0.41ms_(P100_1.77ms_%3C50ms)-brightgreen.svg)](#-empirical-latency-analytics-p50--p70--p100-task-requirement-4)
[![Guardrail](https://img.shields.io/badge/Guardrail_Accuracy-85.71%25_(F1_0.85)-brightgreen.svg)](#-guardrail-precision--recall-benchmark-task-requirement-6)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org)

> High-performance, guardrailed Voice-to-Answer RAG system built for **Hacker House Goa 2026 (Task #2)**.
> **Local query-time pipeline (retrieval + guardrail logic): P50 = 0.6 ms | P100 = 4.7 ms — ~10× under the official sub-50 ms target.**
> **Full end-to-end voice pipeline (real ElevenLabs Scribe v2 STT + Groq LLaMA-3.1 LLM): P50 = 1.42 s (1,416 ms) | P70 = 1.50 s | P100 = 1.53 s — governed by mandated cloud-network roundtrips.**
> **5-layer guardrail suite: 85.71% decision accuracy (Precision 0.778 / Recall 0.933 / F1 0.848) across 35 benchmark queries.**

---

## ⚡ Key Highlights & Architecture

1. **Voice Input (Req #1)**: Powered by **ElevenLabs (`scribe_v2`)** as the primary Speech-to-Text engine. Sarvam AI (`saaras:v3`) remains available as an optional fallback.
2. **Engineered Chunking (Req #2)**: Swappable chunking strategies (Fixed-window with sliding overlap, Recursive sentence-boundary splitter, and Semantic paragraph splitter) evaluated on the complete `ai4bharat/MSMARCO-XI` corpus.
3. **Sub-50ms Vector Retrieval (Req #3)**: ChromaDB manages local disk persistence while query-time retrieval uses a pre-warmed, in-memory term-frequency cosine index built with NumPy (`retrieval/vector_store.py`), delivering **0.41 ms P50** and **1.77 ms P100** retrieval latency after warmup.
4. **5-Layer Defense-in-Depth Guardrails (Req #6)**:
   - **Layer 1 (Pre-Gen)**: Unsafe / Inappropriate Input & Prompt-Injection Blacklist Filter.
   - **Layer 2 (Pre-Gen)**: Insufficient Context Gate (refuses when zero relevant passages are retrieved).
   - **Layer 3 (Pre-Gen)**: Off-Topic Confidence Threshold Gate (`similarity < 0.18` threshold).
   - **Layer 4 (Post-Gen)**: Numeric-fabrication and stemmed lexical-overlap grounding checks.
   - **Layer 5 (Post-Gen)**: Hedge/non-answer detection, which turns model abstentions into proper refusals.
5. **Real Neural Generation**: Powered by **Meta LLaMA 3.1 (`llama-3.1-8b-instant`)** on Groq Cloud LPUs generating fluent, unclipped, factual answers.
6. **Execution Harness & Resilience (Task Requirement #5)**: Typed `BaseStep`/`StepResult` orchestration with exponential backoff and jitter to survive API rate limits under heavy burst load.
7. **Empirical Latency Telemetry (Task Requirement #4)**: Automatic SQLite logging measuring empirical **P50 / P70 / P100** percentiles across real query runs.

---

## 🛡️ 5-Layer Defense-in-Depth Guardrail Suite

The system implements a five-layer safety and grounding architecture that can independently refuse generation or a completed answer:

1. **Layer 1 — Safety & Blacklist Filter (Pre-Gen)**: Scans input text for prompt injection, exploits, or policy-violating patterns inside `GroundingGuardrailStep`.
2. **Layer 2 — Insufficient Context Gate (Pre-Gen)**: Instantly halts execution if vector retrieval returns no relevant context passages.
3. **Layer 3 — Off-Topic Confidence Threshold Gate (Pre-Gen)**: Validates cosine similarity score against threshold (`SIMILARITY_THRESHOLD = 0.18`). Refuses out-of-domain queries in ~0.05ms.
4. **Layer 4 — Post-Generation Hallucination & Faithfulness Checker (Post-Gen)**: Executed in `VoiceRAGOrchestrator` after `LLMGenerationStep`. Every numeric token in the answer must occur in the retrieved context, and stemmed lexical overlap must meet `hallucination_threshold = 0.20`. This catches fabricated quantitative claims while tolerating reasonable inflections and paraphrases.
5. **Layer 5 — Hedge / Non-Answer Detection (Post-Gen)**: Detects abstention wording such as “I do not have information about …” and converts it into a proper refusal (`refused = True`) instead of returning it as an answer.

---

## 📊 Guardrail Precision / Recall Benchmark (Task Requirement #6)

The full pipeline was evaluated across **35 queries**: 30 authentic `ai4bharat/MSMARCO-XI` queries and five out-of-domain refusal cases. Reproduce the evaluation with:

```bash
python benchmarks/guardrail_eval.py
```

| Metric | Value |
| :--- | ---: |
| Total queries evaluated | 35 |
| True positives (correctly answered) | 14 |
| True negatives (correctly refused) | 16 |
| False positives (false answer) | 4 |
| False negatives (false refusal) | 1 |
| Overall accuracy | 85.71% |
| Precision | 0.7778 |
| Recall | 0.9333 |
| F1 score | 0.8485 |

The latest detailed output is stored in `benchmarks/guardrail_eval_results.json`. The remaining false positives are answers whose terms and specifics occur in the retrieved passages; resolving those cases would require semantic entailment judgment beyond the current lexical grounding check.

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

> **Scope & Target Clarification**: The official Task #3 target of **< 50ms** applies specifically to the core local computation pipeline (chunking, vector index search, and guardrail logic). When exercising real speech recognition (ElevenLabs / Sarvam STT) and remote neural generation (Groq LLaMA 3.1), third-party network API delays dominate total latency. Below, we report both benchmark modes honestly and separately.

### 1. Retrieval-Only Pipeline Latency (STT Bypassed)
Benchmarked across **35 diverse queries** (30 authentic queries directly from `ai4bharat/MSMARCO-XI` + 5 out-of-domain guardrail refusal cases) using text override to evaluate internal pipeline performance:

| Pipeline Stage | P50 (Median) | P70 | P100 (Max Worst-Case) | Stage Description |
| :--- | :--- | :--- | :--- | :--- |
| **STT** | `0.02 ms` | `0.02 ms` | `0.05 ms` | Bypassed (Text Override Input) |
| **Vector Retrieval** | `0.41 ms` | `0.42 ms` | `1.77 ms` | In-Memory NumPy TF Cosine Index |
| **Guardrail Gate (Pre-Gen)** | `0.03 ms` | `0.03 ms` | `1.37 ms` | Layers 1–3 Safety & Threshold Gate |
| **LLM Generation (Text Mode)** | **`273.58 ms`** | **`344.68 ms`** | **`637.36 ms`** | Groq LLaMA-3.1 Cloud API |
| **Hallucination Check (Post-Gen)** | `0.16 ms` | `0.18 ms` | `1.52 ms` | Layer 4 Entity Term-Overlap Check |
| **Total End-to-End** | **`274.18 ms`** | **`345.88 ms`** | **`637.93 ms`** | Complete Text Pipeline Run |

### 2. Full End-to-End Latency (Real STT + Real LLM)
Benchmarked across representative **16kHz WAV audio samples** using real **ElevenLabs Scribe v2 STT** and live **Groq Meta LLaMA 3.1** cloud generation:

| Pipeline Stage | P50 (Median) | P70 | P100 (Max Worst-Case) | Stage Description |
| :--- | :--- | :--- | :--- | :--- |
| **STT (ElevenLabs Scribe v2)** | `1159.56 ms` | `1238.24 ms` | `1492.52 ms` | Remote Speech-to-Text API Network Latency |
| **Vector Retrieval** | `0.68 ms` | `1.39 ms` | `3.03 ms` | **In-Memory Vector Search (`< 50ms` Target Met 🏆)** |
| **Guardrail Gate (Pre-Gen)** | `0.06 ms` | `0.06 ms` | `2.87 ms` | Pre-Generation Safety & Threshold Gate |
| **LLM Generation (Groq LLaMA-3.1)** | **`188.65 ms`** | **`248.90 ms`** | **`519.69 ms`** | Groq Cloud LPU Inference Network Latency |
| **Hallucination Check (Post-Gen)** | `0.12 ms` | `0.18 ms` | `3.99 ms` | Layer 4 Entity Term-Overlap Check |
| **Total End-to-End** | **`1416.41 ms`** | **`1498.35 ms`** | **`1525.77 ms`** | **Real Audio & Neural LLM End-to-End** |

### 🔍 Latency & Performance Breakdown

- **Vector Retrieval**: Achieves **0.41–0.68 ms P50** and **1.77–3.03 ms P100**, comfortably satisfying the sub-50ms retrieval constraint (**Target Met 🏆**).
- **Guardrail Efficiency**: Pre-generation safety and threshold checks execute in **0.03 ms P50**. Out-of-domain queries are rejected in sub-millisecond time, avoiding unnecessary LLM call overhead.
- **Third-Party Network Overhead**: Speech recognition (~1.16s P50) and cloud LLM generation (~0.19s–0.27s P50) reflect external web service roundtrips outside local computational control.

---

## 📂 Repository Structure

```text
voice-rag-goa/
├── app.py                      # FastAPI server + live browser demo UI with audio recorder
├── requirements.txt            # Production dependencies
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
│   └── benchmark.py            # Comparative chunking evaluator across 344 passages
│
├── retrieval/                  # Vector search & embeddings
│   ├── __init__.py
│   └── vector_store.py         # Sub-1ms Vector Store (ChromaDB storage + NumPy in-memory TF cosine index)
│
├── stt/                        # Speech-to-Text inference layer
│   ├── __init__.py
│   └── engine.py               # Unified STT engine (ElevenLabs Scribe v2 & Sarvam AI saaras:v3)
│
├── guardrails/                 # Safety & Hallucination gates
│   ├── __init__.py
│   └── threshold_gate.py       # Layers 1–4: safety, retrieval, and grounding checks
│
├── generation/                 # Grounded synthesis layer
│   ├── __init__.py
│   ├── llm.py                  # Groq Meta LLaMA-3.1 Grounded Synthesizer
│   └── prompts.py              # Strict factual RAG prompts
│
├── harness/                    # Orchestration & Execution harness
│   ├── __init__.py
│   ├── base.py                 # BaseStep & StepResult interfaces with telemetry
│   ├── retry.py                # Exponential backoff retry decorator with jitter
│   └── orchestrator.py         # End-to-end Voice-RAG pipeline coordinator
│
├── infrastructure/             # Metrics & Storage persistence
│   ├── __init__.py
│   └── metrics_db.py           # SQLite stage-by-stage latency logging & percentile calculator
│
├── benchmarks/                 # Latency analytics & test suite
│   ├── __init__.py
│   ├── audio_samples/          # Representative WAV audio files for benchmark queries
│   ├── audio_manifest.json     # Audio query manifest
│   ├── generate_audio_samples.py # Script for generating WAV benchmark audio files
│   ├── test_queries.json       # 30 authentic MSMARCO-XI extracted queries
│   ├── guardrail_eval.py       # Precision/recall evaluation across 35 queries
│   ├── diagnose_false_positives.py # Diagnostic report for guardrail false positives
│   └── run_benchmarks.py       # Dual-mode empirical benchmark runner (Retrieval-Only & Full E2E)
│
└── data/                       # Persistent storage (ignored in git)
    ├── chroma/                 # Vector database files (344 indexed passages)
    └── metrics.db              # Latency logs database
```

---

## 🚀 Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Add GROQ_API_KEY and an STT key:
# ELEVENLABS_API_KEY=... (SARVAM_API_KEY is optional fallback)

# 4. Ingest MSMARCO-XI corpus
python -m dataset.loader --samples 30 --strategy recursive_sentence

# 5. Run chunking evaluation and the dual-mode latency benchmark
python -m chunking.benchmark
python benchmarks/run_benchmarks.py
python benchmarks/guardrail_eval.py

# 6. Launch the live demo server
python app.py
```

The demo is served at `http://127.0.0.1:8000`. It accepts live microphone input, uploaded audio, or a text-only query. `GET /api/metrics` returns the latency percentiles recorded in SQLite.

---

## 🔐 Configuration

Copy `.env.example` to `.env` and set the values you need:

| Variable | Purpose |
| :--- | :--- |
| `ELEVENLABS_API_KEY` | Primary STT provider (ElevenLabs Scribe v2). |
| `SARVAM_API_KEY` | Optional STT fallback provider (Sarvam `saaras:v3`). |
| `GROQ_API_KEY` | Enables Groq-backed grounded generation; required for benchmarks. |
| `HF_TOKEN` | Optional Hugging Face token for dataset access. |
| `SIMILARITY_THRESHOLD` | Retrieval confidence required before generation (default: `0.18`). |

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
