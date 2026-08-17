import json
import os
import argparse
from typing import Any
from dotenv import load_dotenv
from datasets import load_dataset
from chunking.strategies import get_chunker
from retrieval.vector_store import get_vector_store
from config.settings import settings
from config.logger import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv(override=True)

DATASET_NAME = "ai4bharat/MSMARCO-XI"
TEST_QUERIES_PATH = "benchmarks/test_queries.json"

def parse_msmarco_record(item: dict[str, Any], record_idx: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parses a single record from ai4bharat/MSMARCO-XI schema:
    - Eng_Query / query
    - Eng_Answer / Answer
    - passages: { is_selected: [...], English_passages: [...], Translated_passages: [...] }
    """
    passages = item.get("passages", {})
    is_selected = passages.get("is_selected", [])
    eng_passages = passages.get("English_passages", [])
    trans_passages = passages.get("Translated_passages", [])

    eng_query = item.get("Eng_Query") or item.get("query", "")
    indic_query = item.get("query", "")
    eng_answer = item.get("Eng_Answer") or item.get("Answer", "")
    query_id = item.get("query_id", record_idx)

    extracted_docs = []

    # Process English + Translated passages
    for p_idx, text in enumerate(eng_passages):
        selected = is_selected[p_idx] if p_idx < len(is_selected) else 0
        trans_text = trans_passages[p_idx] if p_idx < len(trans_passages) else ""
        
        if text and str(text).strip():
            extracted_docs.append({
                "id": f"msmarco_{query_id}_p{p_idx}",
                "text": str(text).strip(),
                "metadata": {
                    "query_id": str(query_id),
                    "eng_query": eng_query,
                    "indic_query": indic_query,
                    "is_selected": int(selected),
                    "target_lang": item.get("target_lang", "unknown")
                }
            })

    test_case = {
        "query_id": str(query_id),
        "eng_query": eng_query,
        "indic_query": indic_query,
        "ground_truth_answer": eng_answer
    }

    return extracted_docs, test_case

def ingest_msmarco_dataset(
    split: str = "validation",
    max_samples: int = 30,
    strategy_name: str = "recursive_sentence"
):
    """
    Streams the official ai4bharat/MSMARCO-XI dataset from Hugging Face using HF_TOKEN,
    applies engineered multi-strategy chunking, and indexes into ChromaDB.
    """
    hf_token = os.getenv("HF_TOKEN") or settings.hf_token or None
    if hf_token and ("your_" in hf_token or "here" in hf_token):
        hf_token = None
    token_status = "Authenticated with HF_TOKEN" if hf_token else "Unauthenticated"
    logger.info(f"Connecting to Hugging Face: {DATASET_NAME} (Split: '{split}', Status: {token_status})...")
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        parquet_file = "validation/hinval.parquet" if split == "validation" else "train/hintrain.parquet"
        logger.info(f"Downloading direct parquet file '{parquet_file}' from HF Hub...")
        local_path = hf_hub_download(repo_id=DATASET_NAME, filename=parquet_file, repo_type="dataset", token=hf_token)
        df = pd.read_parquet(local_path)
        ds = df.to_dict(orient="records")
        logger.info(f"Loaded {len(ds)} records from parquet file.")
    except Exception as e:
        logger.warning(f"Direct parquet load failed ({e}), falling back to load_dataset streaming...")
        ds = load_dataset(
            DATASET_NAME,
            "default",
            split=split,
            streaming=True,
            token=hf_token
        )

    chunker = get_chunker(strategy_name)
    col = get_vector_store()

    all_chunks = []
    ids = []
    metadatas = []
    benchmark_queries = []
    doc_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Streaming & Indexing {max_samples} MSMARCO-XI records...", total=max_samples)

        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            
            docs, test_case = parse_msmarco_record(item, i)
            if test_case["eng_query"]:
                benchmark_queries.append(test_case)

            for doc in docs:
                doc_count += 1
                chunks = chunker.chunk(doc["text"], metadata=doc["metadata"])
                for c_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk["text"])
                    ids.append(f"{doc['id']}_c{c_idx}")
                    metadatas.append(chunk["metadata"])

            progress.update(task, advance=1)

    # Save real dataset test queries for benchmarks
    os.makedirs("benchmarks", exist_ok=True)
    with open(TEST_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_queries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(benchmark_queries)} authentic test queries from MSMARCO-XI to '{TEST_QUERIES_PATH}'.")

    # Upsert into ChromaDB
    if all_chunks:
        logger.info(f"Upserting {len(all_chunks)} chunks into ChromaDB ({col.name})...")
        batch_size = 25
        for b_start in range(0, len(all_chunks), batch_size):
            b_end = b_start + batch_size
            col.upsert(
                documents=all_chunks[b_start:b_end],
                ids=ids[b_start:b_end],
                metadatas=metadatas[b_start:b_end]
            )
        logger.info(f"✅ Successfully indexed {len(all_chunks)} chunks from {doc_count} passages from ai4bharat/MSMARCO-XI!")

def ingest_corpus(strategy_name: str = "recursive_sentence"):
    ingest_msmarco_dataset(split="validation", max_samples=30, strategy_name=strategy_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest official ai4bharat/MSMARCO-XI dataset")
    parser.add_argument("--samples", type=int, default=30, help="Number of records to stream")
    parser.add_argument("--strategy", type=str, default="recursive_sentence", help="Chunking strategy")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (validation or train)")
    args = parser.parse_args()

    ingest_msmarco_dataset(split=args.split, max_samples=args.samples, strategy_name=args.strategy)
