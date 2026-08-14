import os
import sys
import time
import re
import json

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.orchestrator import VoiceRAGOrchestrator
from guardrails.threshold_gate import stem_word
from rich.console import Console
from rich.table import Table

console = Console()

TARGET_QUERY = "What was the closing stock price of Apple on August 12, 1998?"

def compute_overlap_ratio(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0
    ans_raw = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower())) - {
        "this", "that", "from", "with", "have", "were", "they", "their", "about", "would", "which", "there", "these", "where"
    }
    ctx_raw = set(re.findall(r'\b[a-zA-Z]{4,}\b', context.lower()))
    if not ans_raw:
        return 1.0
    ans_words = {stem_word(w) for w in ans_raw}
    ctx_words = {stem_word(w) for w in ctx_raw}
    return len(ans_words.intersection(ctx_words)) / len(ans_words)

def run_determinism_check(num_runs: int = 10, pacing_delay: float = 3.5):
    console.print(f"\n[cyan]🔍 Running Determinism Check across {num_runs} consecutive runs...[/cyan]")
    console.print(f"[dim]Query: '{TARGET_QUERY}' | Pacing: {pacing_delay}s between calls[/dim]\n")

    orchestrator = VoiceRAGOrchestrator()

    # Pre-retrieve context for exact overlap computation
    ret_res = orchestrator.retrieval.run({"transcript": TARGET_QUERY})
    ctx_docs = ret_res.data.get("documents", [])
    context_text = "\n".join(ctx_docs)

    table = Table(title="🎯 Hallucination Check Determinism Analysis")
    table.add_column("Run #", justify="right", style="dim")
    table.add_column("Refused?", justify="center")
    table.add_column("Overlap Ratio", justify="right", style="cyan")
    table.add_column("Layer 4 Pass?", justify="center")
    table.add_column("Generated Answer Snippet", style="white")

    outcomes = []
    ratios = []

    for i in range(1, num_runs + 1):
        res = orchestrator.process(text_override=TARGET_QUERY)
        refused = res.get("refused", False)
        ans = res.get("answer", "")
        reason = res.get("refusal_reason", "")
        
        # Calculate overlap ratio against context
        ratio = compute_overlap_ratio(ans, context_text)
        ratios.append(ratio)

        action_str = "[bold red]REFUSED[/bold red]" if refused else "[bold green]ANSWERED[/bold green]"
        layer4_pass = "No ❌" if refused else "Yes ✅"
        outcomes.append("REFUSED" if refused else "ANSWERED")

        ans_snippet = ans[:70].replace("\n", " ") + ("..." if len(ans) > 70 else "")
        table.add_row(str(i), action_str, f"{ratio:.4f}", layer4_pass, ans_snippet)

        if i < num_runs:
            time.sleep(pacing_delay)

    console.print(table)

    refused_cnt = outcomes.count("REFUSED")
    answered_cnt = outcomes.count("ANSWERED")
    flip_cnt = sum(1 for a, b in zip(outcomes, outcomes[1:]) if a != b)

    console.print("\n[bold yellow]📊 Summary Statistics:[/bold yellow]")
    console.print(f" • Total Runs: {num_runs}")
    console.print(f" • Refused Count: [red]{refused_cnt}[/red]")
    console.print(f" • Answered Count: [green]{answered_cnt}[/green]")
    console.print(f" • Outcome Flips between consecutive runs: [bold magenta]{flip_cnt}[/bold magenta]")
    console.print(f" • Overlap Ratio Range: [{min(ratios):.4f} - {max(ratios):.4f}]")
    console.print(f" • Overlap Ratio Mean: {sum(ratios)/len(ratios):.4f}\n")

    return outcomes, ratios

if __name__ == "__main__":
    run_determinism_check()
