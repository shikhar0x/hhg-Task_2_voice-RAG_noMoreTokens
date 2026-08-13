import json
import os
from chunking.strategies import get_chunker
from retrieval.vector_store import get_vector_store
from config.logger import logger

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Hacker House Goa Overview",
        "text": "Hacker House Goa is an elite 4-day builder residency running from October 28 to 31, 2026 in Goa, India. It gathers 247 top engineers, founders, and creators to launch breakthrough products on the beach with ultra-high-speed fiber, mentorship, and VC backing."
    },
    {
        "id": "doc_2",
        "title": "Goa Capital and Official Language",
        "text": "Panaji is the state capital of Goa, situated on the banks of the Mandovi river. Konkani is the official state language of Goa, written in the Devanagari script, while Marathi and English are widely spoken across all administrative and commercial domains."
    },
    {
        "id": "doc_3",
        "title": "Task 2 Requirements HH Goa",
        "text": "Task #2 of HH Goa requires building a Voice-Enabled RAG pipeline. Key requirements include: Sarvam or ElevenLabs STT, multi-strategy chunking, P50/P70/P100 latency analytics, structured harness orchestration with retries, and strict guardrails that know when not to answer."
    },
    {
        "id": "doc_4",
        "title": "Sarvam AI Multilingual Models",
        "text": "Sarvam AI is an Indian foundational AI company building speech and language models for Indic languages. Their Saarika v2 model offers fast speech-to-text transcription across 10 Indian languages and Indian English with high acoustic resilience."
    },
    {
        "id": "doc_5",
        "title": "Vector Embeddings and Cosine Distance",
        "text": "In vector search systems, cosine distance measures the angular difference between high-dimensional document vectors. A cosine distance of 0 denotes identical direction (similarity = 1.0), whereas distances greater than 0.35 indicate weak contextual affinity."
    }
]

def ingest_corpus(strategy_name: str = "recursive_sentence", custom_docs: list[dict] | None = None):
    docs = custom_docs or SAMPLE_DOCUMENTS
    chunker = get_chunker(strategy_name)
    col = get_vector_store()

    all_chunks = []
    ids = []
    metadatas = []

    logger.info(f"Ingesting {len(docs)} documents using chunking strategy '{strategy_name}'...")
    chunk_counter = 0

    for doc in docs:
        chunks = chunker.chunk(doc["text"], metadata={"doc_id": doc["id"], "title": doc.get("title", "")})
        for c in chunks:
            chunk_counter += 1
            all_chunks.append(c["text"])
            ids.append(f"{doc['id']}_chk_{chunk_counter}")
            metadatas.append(c["metadata"])

    if all_chunks:
        col.upsert(documents=all_chunks, ids=ids, metadatas=metadatas)
        logger.info(f"Successfully indexed {len(all_chunks)} chunks into ChromaDB collection '{col.name}'.")

if __name__ == "__main__":
    ingest_corpus()
