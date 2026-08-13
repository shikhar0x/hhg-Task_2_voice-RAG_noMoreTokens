import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.vector_store import VectorRetrievalStep

OUT_OF_DOMAIN_QUERIES = [
    {"query": "What is the recipe for baking a chocolate lava cake?", "type": "Out-of-Domain Refusal", "ground_truth_answer": "No Answer Present."},
    {"query": "How do quantum computers factor 2048-bit RSA keys using Shor's algorithm?", "type": "Out-of-Domain Refusal", "ground_truth_answer": "No Answer Present."},
    {"query": "What was the closing stock price of Apple on August 12, 1998?", "type": "Out-of-Domain Refusal", "ground_truth_answer": "No Answer Present."},
    {"query": "Who won the FIFA World Cup in 1930 in Uruguay?", "type": "Out-of-Domain Refusal", "ground_truth_answer": "No Answer Present."},
    {"query": "How do I build a nuclear fusion reactor at home?", "type": "Out-of-Domain Refusal", "ground_truth_answer": "No Answer Present."}
]

def main():
    retrieval_step = VectorRetrievalStep()

    queries_data = []
    if os.path.exists("benchmarks/test_queries.json"):
        with open("benchmarks/test_queries.json", "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
            for item in raw_cases:
                q = item.get("eng_query") or item.get("indic_query")
                gt = item.get("ground_truth_answer", "").strip()
                if q:
                    queries_data.append({
                        "query_id": item.get("query_id", "N/A"),
                        "query": q,
                        "ground_truth_answer": gt,
                        "is_in_domain": gt != "No Answer Present." and bool(gt)
                    })

    for idx, item in enumerate(OUT_OF_DOMAIN_QUERIES, start=1):
        queries_data.append({
            "query_id": f"OOD_{idx:02d}",
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "is_in_domain": False
        })

    results = []
    for item in queries_data:
        res = retrieval_step.run({"transcript": item["query"]})
        top_sim = res.data.get("top_similarity", 0.0) if res.success else 0.0
        results.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "top_similarity": top_sim,
            "is_in_domain": item["is_in_domain"],
            "ground_truth": item["ground_truth_answer"]
        })

    results.sort(key=lambda x: x["top_similarity"], reverse=True)

    print("=" * 105)
    print(f"{'#':<3} | {'Query ID':<10} | {'Similarity':<10} | {'Domain Status':<25} | {'Query Text':<50}")
    print("=" * 105)

    for idx, r in enumerate(results, start=1):
        status = "True In-Domain (Answer)" if r["is_in_domain"] else "True Out-of-Domain"
        q_text = r["query"][:48] + (".." if len(r["query"]) > 48 else "")
        print(f"{idx:<3} | {r['query_id']:<10} | {r['top_similarity']:<10.4f} | {status:<25} | {q_text:<50}")

    print("=" * 105)

    in_domain_sims = [r["top_similarity"] for r in results if r["is_in_domain"]]
    ood_sims = [r["top_similarity"] for r in results if not r["is_in_domain"]]

    print(f"\nTotal Queries Evaluated: {len(results)} ({len(in_domain_sims)} In-Domain, {len(ood_sims)} Out-of-Domain)")
    print(f"Minimum Similarity for True In-Domain Queries: {min(in_domain_sims):.4f}")
    print(f"Maximum Similarity for True Out-of-Domain Queries: {max(ood_sims):.4f}")
    print(f"Scores for True In-Domain Queries: {sorted(in_domain_sims)}")

if __name__ == "__main__":
    main()
