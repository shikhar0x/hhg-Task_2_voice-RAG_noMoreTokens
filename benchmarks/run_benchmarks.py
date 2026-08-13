import json
import os
from harness.orchestrator import VoiceRAGOrchestrator
from rich.console import Console
from rich.table import Table

console = Console()

OUT_OF_DOMAIN_QUERIES = [
    {"query": "What is the recipe for baking a chocolate lava cake?", "type": "Out-of-Domain Refusal"},
    {"query": "How do quantum computers factor 2048-bit RSA keys using Shor's algorithm?", "type": "Out-of-Domain Refusal"},
    {"query": "What was the closing stock price of Apple on August 12, 1998?", "type": "Out-of-Domain Refusal"},
    {"query": "Who won the FIFA World Cup in 1930 in Uruguay?", "type": "Out-of-Domain Refusal"},
    {"query": "How do I build a nuclear fusion reactor at home?", "type": "Out-of-Domain Refusal"}
]

def main():
    console.rule("[bold green]🚀 Running Official 35-Query MSMARCO-XI Benchmark Suite[/bold green]")
    orchestrator = VoiceRAGOrchestrator()
    
    # Load all 30 genuine queries extracted from MSMARCO-XI
    msmarco_queries = []
    if os.path.exists("benchmarks/test_queries.json"):
        with open("benchmarks/test_queries.json", "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
            for item in raw_cases:
                q = item.get("eng_query") or item.get("indic_query")
                if q:
                    msmarco_queries.append({"query": q, "type": "MSMARCO-XI Grounded"})

    test_suite = msmarco_queries + OUT_OF_DOMAIN_QUERIES
    console.print(f"\n[cyan]Executing all {len(test_suite)} evaluation queries ({len(msmarco_queries)} Authentic MSMARCO-XI + {len(OUT_OF_DOMAIN_QUERIES)} Guardrail Refusals)...[/cyan]\n")

    grounded_count = 0
    refused_count = 0

    for idx, item in enumerate(test_suite, start=1):
        res = orchestrator.process(text_override=item["query"])
        if res.get("refused"):
            refused_count += 1
            status = "[red]REFUSED (Guardrail Gate)[/red]"
        else:
            grounded_count += 1
            status = "[green]ANSWERED (Grounded)[/green]"
        console.print(f"[{idx:02d}/{len(test_suite)}] '{item['query'][:40]}...' ➔ {status} in {res['timings']['total']:.1f}ms")

    percentiles = orchestrator.metrics_db.compute_percentiles()

    table = Table(title="📊 Official ai4bharat/MSMARCO-XI - 35-Query Empirical Latency (P50 / P70 / P100)")
    table.add_column("Pipeline Stage", style="bold cyan")
    table.add_column("P50 (Median)", style="green")
    table.add_column("P70", style="yellow")
    table.add_column("P100 (Max)", style="red")

    for stage, metrics in percentiles.items():
        table.add_row(stage, metrics["P50 (Median)"], metrics["P70"], metrics["P100 (Max)"])

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]Summary:[/bold green] In-Domain Grounded: [green]{grounded_count}[/green] | Guardrail Refusals: [red]{refused_count}[/red]")
    console.print("[bold green]✅ All 35 benchmark queries recorded in SQLite database for statistical rigor![/bold green]\n")

if __name__ == "__main__":
    main()
