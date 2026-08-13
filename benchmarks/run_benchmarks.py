import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
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

def check_env_keys():
    load_dotenv(override=True)
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.strip() in ("", "your_groq_api_key_here", "gsk_your_groq_api_key_here"):
        console.print("\n[bold red]❌ FATAL BENCHMARK ERROR: GROQ_API_KEY is missing or unconfigured in .env![/bold red]")
        console.print("[red]The official benchmark suite requires a valid GROQ_API_KEY to measure real LLM generation latency.[/red]")
        console.print("[red]Benchmarking with non-LLM fallbacks produces invalid latency metrics. Execution aborted.[/red]\n")
        sys.exit(1)

    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not eleven_key and not sarvam_key:
        console.print("\n[bold red]❌ FATAL BENCHMARK ERROR: Neither ELEVENLABS_API_KEY nor SARVAM_API_KEY is set in .env![/bold red]")
        console.print("[red]The full end-to-end benchmark mode requires a valid Speech-to-Text API key. Execution aborted.[/red]\n")
        sys.exit(1)

def run_retrieval_only_benchmark(orchestrator, test_suite):
    console.rule("[bold cyan]📊 Mode 1: Retrieval-Only Latency (STT bypassed)[/bold cyan]")
    console.print(f"\n[cyan]Executing all {len(test_suite)} queries in text-override mode (STT bypassed)...[/cyan]\n")

    grounded_count = 0
    refused_count = 0

    for idx, item in enumerate(test_suite, start=1):
        res = orchestrator.process(text_override=item["query"], mode="retrieval_only")
        if res.get("refused"):
            refused_count += 1
            status = "[red]REFUSED (Guardrail Gate)[/red]"
        else:
            grounded_count += 1
            status = "[green]ANSWERED (Grounded LLaMA-3.1)[/green]"
        console.print(f"[{idx:02d}/{len(test_suite)}] '{item['query'][:38]}...' ➔ {status} in {res['timings']['total']:.1f}ms")
        # Rate Limit Pacing Rationale: Groq llama-3.1-8b-instant enforces 14,400 TPM (~240 tokens/sec).
        # Multi-passage RAG prompts average ~500 tokens. A 0.8s pacing delay (~625 tokens/sec window)
        # prevents bursting past Groq's TPM sliding rate limit, eliminating HTTP 429 backoff tail latency.
        time.sleep(0.8)

    percentiles = orchestrator.metrics_db.compute_percentiles(mode="retrieval_only")
    return percentiles, grounded_count, refused_count

def run_full_e2e_benchmark(orchestrator):
    console.rule("[bold green]🚀 Mode 2: Full End-to-End Latency (Real STT + Groq LLaMA-3.1)[/bold green]")
    time.sleep(2.0)  # Pacing pause to ensure Groq rate limit window resets

    manifest_path = "benchmarks/audio_manifest.json"
    audio_queries = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            audio_queries = json.load(f)

    if not audio_queries:
        console.print("[bold red]❌ No audio queries found in benchmarks/audio_manifest.json![/bold red]")
        sys.exit(1)

    console.print(f"\n[cyan]Executing {len(audio_queries)} representative queries using real WAV audio files through ElevenLabs/Sarvam STT & Groq LLM...[/cyan]\n")

    grounded_count = 0
    refused_count = 0

    for idx, item in enumerate(audio_queries, start=1):
        audio_path = item.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            console.print(f"[yellow]Skipping missing audio file: {audio_path}[/yellow]")
            continue

        res = orchestrator.process(audio_path=audio_path, mode="end_to_end")

        if not res.get("refused") and res.get("generation_provider") == "clean_fallback":
            console.print(f"\n[bold red]❌ FATAL ERROR: Groq generation fallback detected for query '{item['query']}'![/bold red]")
            console.print("[red]Ensure GROQ_API_KEY is valid. Falling back to non-LLM response is prohibited in benchmarks.[/red]\n")
            sys.exit(1)

        if res.get("refused"):
            refused_count += 1
            status = "[red]REFUSED (Guardrail Gate)[/red]"
        else:
            grounded_count += 1
            status = f"[green]ANSWERED via {res.get('stt_engine', 'STT')} + Groq LLaMA-3.1[/green]"

        stt_dur = res['timings'].get('stt', 0.0)
        total_dur = res['timings'].get('total', 0.0)
        transcript_snippet = res.get('transcript', item['query'])[:38]
        console.print(f"[{idx:02d}/{len(audio_queries)}] '{transcript_snippet}...' ➔ {status} (STT: {stt_dur:.1f}ms | Total: {total_dur:.1f}ms)")
        time.sleep(0.5)

    percentiles = orchestrator.metrics_db.compute_percentiles(mode="end_to_end")
    return percentiles, grounded_count, refused_count

def print_summary_table(title: str, percentiles: dict[str, dict[str, str]]):
    table = Table(title=title)
    table.add_column("Pipeline Stage", style="bold cyan")
    table.add_column("P50 (Median)", style="green")
    table.add_column("P70", style="yellow")
    table.add_column("P100 (Max)", style="red")

    for stage, metrics in percentiles.items():
        table.add_row(stage, metrics["P50 (Median)"], metrics["P70"], metrics["P100 (Max)"])

    console.print("\n")
    console.print(table)

def main():
    console.rule("[bold green]🚀 Voice-RAG Benchmark Suite (Dual-Mode Verification)[/bold green]")
    check_env_keys()

    orchestrator = VoiceRAGOrchestrator()

    # Clear previous latency logs for clean benchmark percentiles
    import sqlite3
    with sqlite3.connect(orchestrator.metrics_db.db_path) as conn:
        conn.execute("DELETE FROM latency_logs")
        conn.commit()

    # Load 30 genuine queries extracted from MSMARCO-XI
    msmarco_queries = []
    if os.path.exists("benchmarks/test_queries.json"):
        with open("benchmarks/test_queries.json", "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
            for item in raw_cases:
                q = item.get("eng_query") or item.get("indic_query")
                if q:
                    msmarco_queries.append({"query": q, "type": "MSMARCO-XI Grounded"})

    text_suite = msmarco_queries + OUT_OF_DOMAIN_QUERIES

    # 1. Retrieval-Only Benchmark (STT Bypassed)
    retrieval_percentiles, ret_grounded, ret_refused = run_retrieval_only_benchmark(orchestrator, text_suite)

    # 2. Full End-to-End Benchmark (Real STT + Groq LLM)
    e2e_percentiles, e2e_grounded, e2e_refused = run_full_e2e_benchmark(orchestrator)

    # Display Tables & Summaries
    console.rule("[bold yellow]📊 Benchmark Empirical Latency Summaries[/bold yellow]")
    
    print_summary_table(
        "📊 Retrieval-Only Latency (STT bypassed) - 35-Query Empirical Latency (P50 / P70 / P100)",
        retrieval_percentiles
    )
    console.print(f"[bold green]Mode 1 Summary:[/bold green] In-Domain Grounded: [green]{ret_grounded}[/green] | Guardrail Refusals: [red]{ret_refused}[/red]")

    print_summary_table(
        "📊 Full End-to-End Latency (Real STT + Groq LLaMA-3.1) - Audio Sample Empirical Latency (P50 / P70 / P100)",
        e2e_percentiles
    )
    console.print(f"[bold green]Mode 2 Summary:[/bold green] In-Domain Grounded: [green]{e2e_grounded}[/green] | Guardrail Refusals: [red]{e2e_refused}[/red]")

    console.print("\n[bold green]✅ Both Retrieval-Only and Full End-to-End benchmarks completed and recorded separately![/bold green]\n")

if __name__ == "__main__":
    main()
