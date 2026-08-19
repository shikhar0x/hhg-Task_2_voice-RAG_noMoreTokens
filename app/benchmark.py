"""Dashboard + CLI wrapper around the official fast-path window.

`total (ms)` is retrieve + guardrail + extract (`generate=false`), budget 50ms.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import LATENCY_BUDGET_MS
from benchmarks.fastpath import run_fastpath

console = Console()


def run_benchmark(n: int = 80, verbose: bool = True, orch=None) -> dict:
    results = run_fastpath(n=n, warmup=max(5, min(10, n // 8)), orch=orch)

    if verbose:
        fp = results["fast_path_ms"]
        table = Table(
            show_header=True,
            header_style="bold dim white",
            box=box.SIMPLE_HEAD,
            padding=(0, 4),
            expand=True,
        )
        table.add_column("", justify="left")
        table.add_column("AVG", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P70", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P100", justify="right")
        table.add_row(
            "fast path (ms)",
            f"{fp['avg']:.2f}",
            f"{fp['p50']:.2f}",
            f"{fp['p70']:.2f}",
            f"{fp['p95']:.1f}",
            f"{fp['p100']:.2f}",
        )
        badge = results["badge_text"]
        color = "#15803d" if results["status"] == "PASS" else "#b91c1c"
        console.print(
            Panel(
                table,
                title="[bold]Fast-path latency[/bold]  (transcript → extractive, no LLM)",
                subtitle=f"[bold white on {color}]  {badge}  [/]",
                border_style="gray23",
                box=box.ROUNDED,
            )
        )

    results.setdefault("budget_ms", LATENCY_BUDGET_MS)
    return results


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    results = run_benchmark(n=n, verbose=True)
    if results["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
