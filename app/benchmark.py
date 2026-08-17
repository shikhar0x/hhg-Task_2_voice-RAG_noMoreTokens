"""Measure end-to-end retrieval latency (embed + FAISS search) against the
50ms budget defined in app/config.py.

Usage:
    python -m app.benchmark [n_queries]
"""
import os
import statistics
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import LATENCY_BUDGET_MS
from app.retriever import search, warmup

console = Console()

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def run_benchmark(n: int = 50, verbose: bool = True) -> dict:
    if verbose:
        console.print("[dim]Warming up (model load + first inference)...[/dim]")
    warmup()

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = search(query, top_k=5)
        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)

    metrics = {}
    for name, values in [("embed (ms)", embed_ms), ("search (ms)", search_ms), ("total (ms)", total_ms)]:
        metrics[name] = {
            "avg": round(statistics.mean(values), 2),
            "p50": round(percentile(values, 50), 2),
            "p95": round(percentile(values, 95), 2),
            "p99": round(percentile(values, 99), 2),
        }

    p95_total = metrics["total (ms)"]["p95"]
    is_pass = p95_total <= LATENCY_BUDGET_MS

    # Format p95 string cleanly (e.g. 6.1 instead of 6.10 if single decimal)
    p95_str = f"{p95_total:.1f}" if round(p95_total, 1) == p95_total else f"{p95_total:.2f}"
    if is_pass:
        badge_text = f"PASS -- p95 {p95_str}ms within budget"
    else:
        badge_text = f"FAIL -- p95 {p95_str}ms over budget"

    results = {
        "status": "PASS" if is_pass else "FAIL",
        "p95_total": p95_total,
        "budget_ms": LATENCY_BUDGET_MS,
        "badge_text": badge_text,
        "metrics": metrics,
        "n_queries": n
    }

    if verbose:
        header_grid = Table.grid(expand=True)
        header_grid.add_column(justify="left")
        header_grid.add_column(justify="right")
        header_grid.add_row(
            "[bold white]Latency benchmark[/bold white]",
            "[bold white on #6366f1]  Run benchmark  [/bold white on #6366f1]"
        )

        table = Table(show_header=True, header_style="bold dim white", box=box.SIMPLE_HEAD, padding=(0, 4), expand=True)
        table.add_column("", justify="left", style="white")
        table.add_column("AVG", justify="right", style="bold white")
        table.add_column("P50", justify="right", style="bold white")
        table.add_column("P95", justify="right", style="bold white")
        table.add_column("P99", justify="right", style="bold white")

        for stage, data in metrics.items():
            table.add_row(
                stage,
                f"{data['avg']:.2f}",
                f"{data['p50']:.2f}",
                f"{data['p95']:.1f}" if round(data['p95'], 1) == data['p95'] else f"{data['p95']:.2f}",
                f"{data['p99']:.2f}"
            )

        if is_pass:
            badge_render = f"[bold white on #15803d]  {badge_text}  [/bold white on #15803d]"
        else:
            badge_render = f"[bold white on #b91c1c]  {badge_text}  [/bold white on #b91c1c]"

        content_grid = Table.grid(expand=True)
        content_grid.add_row(header_grid)
        content_grid.add_row("")
        content_grid.add_row(table)
        content_grid.add_row("")
        content_grid.add_row(badge_render)

        panel = Panel(
            content_grid,
            border_style="gray23",
            box=box.ROUNDED,
            expand=False,
            padding=(1, 2)
        )
        console.print(panel)

    return results


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    results = run_benchmark(n=n, verbose=True)
    if results["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

