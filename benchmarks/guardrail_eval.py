import os
import sys
import json
import time

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.orchestrator import VoiceRAGOrchestrator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

OUT_OF_DOMAIN_QUERIES = [
    {"query_id": "OOD_01", "query": "What is the recipe for baking a chocolate lava cake?", "ground_truth_answer": "No Answer Present."},
    {"query_id": "OOD_02", "query": "How do quantum computers factor 2048-bit RSA keys using Shor's algorithm?", "ground_truth_answer": "No Answer Present."},
    {"query_id": "OOD_03", "query": "What was the closing stock price of Apple on August 12, 1998?", "ground_truth_answer": "No Answer Present."},
    {"query_id": "OOD_04", "query": "Who won the FIFA World Cup in 1930 in Uruguay?", "ground_truth_answer": "No Answer Present."},
    {"query_id": "OOD_05", "query": "How do I build a nuclear fusion reactor at home?", "ground_truth_answer": "No Answer Present."}
]

def parse_refusal_layer(reason: str) -> str:
    if not reason:
        return "N/A"
    reason_lower = reason.lower()
    if "safety" in reason_lower or "unsafe" in reason_lower:
        return "Layer 1 (Safety)"
    if "zero relevant passages" in reason_lower or "insufficient context" in reason_lower:
        return "Layer 2 (Zero Context)"
    if "below strict threshold" in reason_lower or "out of domain" in reason_lower:
        return "Layer 3 (Threshold Gate)"
    if "post-generation grounding check" in reason_lower or "hallucination" in reason_lower:
        return "Layer 4 (Post-Gen Grounding)"
    return "Refused (Other)"

def main():
    console.print(Panel.fit("🔍 [bold cyan]Voice-RAG Guardrail Precision / Recall Evaluation[/bold cyan]\nEvaluating 35 benchmark queries through full pipeline...", border_style="cyan"))

    orchestrator = VoiceRAGOrchestrator()

    queries_data = []
    if os.path.exists("benchmarks/test_queries.json"):
        with open("benchmarks/test_queries.json", "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
            for item in raw_cases:
                q = item.get("eng_query") or item.get("indic_query")
                gt = item.get("ground_truth_answer", "").strip()
                if q:
                    should_answer = gt != "No Answer Present." and bool(gt)
                    queries_data.append({
                        "query_id": item.get("query_id", "N/A"),
                        "query": q,
                        "ground_truth_answer": gt,
                        "expected_action": "ANSWER" if should_answer else "REFUSE"
                    })

    for item in OUT_OF_DOMAIN_QUERIES:
        queries_data.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "expected_action": "REFUSE"
        })

    eval_results = []
    tp, tn, fp, fn = 0, 0, 0, 0
    fn_cases = []

    for idx, item in enumerate(queries_data, start=1):
        q_text = item["query"]
        expected = item["expected_action"]

        res = orchestrator.process(text_override=q_text)

        actual_refused = res.get("refused", False)
        actual_action = "REFUSE" if actual_refused else "ANSWER"
        refusal_reason = res.get("refusal_reason", "")
        trigger_layer = parse_refusal_layer(refusal_reason) if actual_refused else "N/A"
        answer = res.get("answer", "")
        sims = res.get("similarities", [])
        top_similarity = sims[0] if sims else 0.0

        # Classification
        if expected == "ANSWER" and actual_action == "ANSWER":
            status = "TP"
            tp += 1
        elif expected == "REFUSE" and actual_action == "REFUSE":
            status = "TN"
            tn += 1
        elif expected == "REFUSE" and actual_action == "ANSWER":
            status = "FP"
            fp += 1
        else:  # expected == "ANSWER" and actual_action == "REFUSE"
            status = "FN"
            fn += 1
            fn_cases.append({
                "query_id": item["query_id"],
                "query": q_text,
                "top_similarity": top_similarity,
                "trigger_layer": trigger_layer,
                "refusal_reason": refusal_reason
            })

        record = {
            "index": idx,
            "query_id": item["query_id"],
            "query": q_text,
            "ground_truth_answer": item["ground_truth_answer"],
            "expected_action": expected,
            "actual_action": actual_action,
            "status": status,
            "refused": actual_refused,
            "refusal_reason": refusal_reason,
            "trigger_layer": trigger_layer,
            "top_similarity": top_similarity,
            "answer": answer,
            "timings": res.get("timings", {})
        }
        eval_results.append(record)

    total = len(eval_results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print Results Table
    table = Table(title="📊 Guardrail Evaluation Query Breakdown")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Query ID", style="cyan")
    table.add_column("Query Text", style="white")
    table.add_column("Expected", justify="center")
    table.add_column("Actual", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Trigger Layer / Reason", style="dim")

    for r in eval_results:
        st = r["status"]
        if st == "TP":
            st_fmt = "[bold green]TP (Correct Answer)[/bold green]"
        elif st == "TN":
            st_fmt = "[bold blue]TN (Correct Refusal)[/bold blue]"
        elif st == "FP":
            st_fmt = "[bold magenta]FP (False Answer)[/bold magenta]"
        else:
            st_fmt = "[bold red]FN (False Refusal)[/bold red]"

        table.add_row(
            str(r["index"]),
            r["query_id"],
            r["query"][:40] + ("..." if len(r["query"]) > 40 else ""),
            r["expected_action"],
            r["actual_action"],
            st_fmt,
            r["trigger_layer"] if r["refused"] else "-"
        )

    console.print("\n")
    console.print(table)

    # Print Metrics Summary
    metrics_table = Table(title="📈 Guardrail Precision / Recall & Confusion Matrix Summary")
    metrics_table.add_column("Metric", style="bold yellow")
    metrics_table.add_column("Count / Value", style="bold white", justify="right")

    metrics_table.add_row("Total Queries Evaluated", str(total))
    metrics_table.add_row("True Positives (TP - Correctly Answered)", f"[green]{tp}[/green]")
    metrics_table.add_row("True Negatives (TN - Correctly Refused)", f"[blue]{tn}[/blue]")
    metrics_table.add_row("False Positives (FP - False Answer)", f"[magenta]{fp}[/magenta]")
    metrics_table.add_row("False Negatives (FN - False Refusal)", f"[red]{fn}[/red]")
    metrics_table.add_row("Overall Accuracy", f"[bold green]{accuracy * 100:.2f}%[/bold green]")
    metrics_table.add_row("Precision", f"{precision:.4f}")
    metrics_table.add_row("Recall", f"{recall:.4f}")
    metrics_table.add_row("F1 Score", f"{f1:.4f}")

    console.print("\n")
    console.print(metrics_table)

    # Call out False Negative cases
    if fn_cases:
        console.print(f"\n[bold red]⚠️ False Negative Refusals Callout ({len(fn_cases)} cases):[/bold red]")
        for fn_item in fn_cases:
            console.print(f" • [bold white]'{fn_item['query']}'[/bold white] (Query ID: {fn_item['query_id']})")
            console.print(f"   - Top Similarity: [cyan]{fn_item['top_similarity']:.4f}[/cyan]")
            console.print(f"   - Triggered Layer: [yellow]{fn_item['trigger_layer']}[/yellow]")
            console.print(f"   - Refusal Reason: {fn_item['refusal_reason']}")
    else:
        console.print("\n[bold green]🎉 Zero False Negative Refusals! All answerable queries were correctly answered.[/bold green]")

    # Save to json file
    output_path = "benchmarks/guardrail_eval_results.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "false_negatives": fn_cases,
        "detailed_results": eval_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    console.print(f"\n[dim]Saved evaluation results to [bold]{output_path}[/bold][/dim]\n")

if __name__ == "__main__":
    main()
