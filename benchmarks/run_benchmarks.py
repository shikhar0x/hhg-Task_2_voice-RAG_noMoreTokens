import json
from harness.orchestrator import VoiceRAGOrchestrator
from dataset.loader import ingest_corpus
from rich.console import Console
from rich.table import Table

console = Console()

TEST_QUERIES = [
    # In-domain grounded queries
    {"query": "When and where is Hacker House Goa taking place?", "expected_refusal": False},
    {"query": "What is the official state language of Goa?", "expected_refusal": False},
    {"query": "What are the core technical requirements for Task 2?", "expected_refusal": False},
    {"query": "What language support does Sarvam Saarika v2 provide?", "expected_refusal": False},
    {"query": "How is cosine distance related to similarity in vector search?", "expected_refusal": False},

    # Out-of-domain / Guardrail refusal queries
    {"query": "What is the recipe for baking a chocolate lava cake?", "expected_refusal": True},
    {"query": "How do quantum computers factor 2048-bit RSA keys?", "expected_refusal": True},
    {"query": "Tell me about the stock price of Apple in 1998.", "expected_refusal": True}
]

def main():
    console.rule("[bold green]🚀 Initializing Corpus & Running Benchmark Suite[/bold green]")
    ingest_corpus(strategy_name="recursive_sentence")

    orchestrator = VoiceRAGOrchestrator()
    console.print(f"\n[cyan]Running {len(TEST_QUERIES)} benchmark queries through full harness...[/cyan]\n")

    for i, item in enumerate(TEST_QUERIES, start=1):
        res = orchestrator.process(text_override=item["query"])
        status = "[red]REFUSED[/red]" if res.get("refused") else "[green]ANSWERED[/green]"
        console.print(f"[{i}/{len(TEST_QUERIES)}] Query: '{item['query'][:40]}...' -> {status} in {res['timings']['total']:.2f}ms")

    percentiles = orchestrator.metrics_db.compute_percentiles()

    table = Table(title="📊 HH Goa Task #2 - Empirical Latency Analytics (P50 / P70 / P100)")
    table.add_column("Pipeline Stage", style="bold cyan")
    table.add_column("P50 (Median)", style="green")
    table.add_column("P70", style="yellow")
    table.add_column("P100 (Max)", style="red")

    for stage, metrics in percentiles.items():
        table.add_row(stage, metrics["P50 (Median)"], metrics["P70"], metrics["P100 (Max)"])

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]✅ Benchmarks completed and recorded in metrics SQLite database![/bold green]\n")

if __name__ == "__main__":
    main()
