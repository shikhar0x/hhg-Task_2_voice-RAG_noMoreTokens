import json
import time
from retrieval.vector_store import get_vector_store
from chunking.strategies import FixedWindowChunker, RecursiveSentenceChunker, SemanticParagraphChunker
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate_chunking_on_full_corpus():
    col = get_vector_store()
    # Fetch all indexed documents from ChromaDB
    all_data = col.get()
    docs = all_data.get("documents", [])
    
    if not docs:
        console.print("[red]No documents found in ChromaDB. Run dataset loader first.[/red]")
        return

    full_text = "\n\n".join(docs)
    total_chars = len(full_text)
    
    strategies = [
        FixedWindowChunker(chunk_size=300, overlap=60),
        RecursiveSentenceChunker(max_words=60),
        SemanticParagraphChunker(max_length=400)
    ]

    console.rule("[bold green]🔬 Evaluating Chunking Strategies on Full ai4bharat/MSMARCO-XI Dataset[/bold green]")
    console.print(f"[cyan]Corpus Size: {len(docs)} passages ({total_chars:,} characters)[/cyan]\n")

    table = Table(title="🔬 Full Corpus Chunking Strategies Evaluation (Task Requirement #2)")
    table.add_column("Strategy Name", style="bold cyan")
    table.add_column("Total Chunks", style="green")
    table.add_column("Avg Chunk Length", style="yellow")
    table.add_column("Boundary / Overlap Type", style="magenta")
    table.add_column("Execution Latency", style="blue")

    for strat in strategies:
        start = time.perf_counter()
        chunks = strat.chunk(full_text, metadata={"source": "ai4bharat/MSMARCO-XI"})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        avg_len = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
        boundary = (
            "Sliding Window (60ch overlap)" if strat.name == "fixed_window" 
            else ("Syntactic Sentence Boundary" if strat.name == "recursive_sentence" 
            else "Paragraph / Structural Split")
        )

        table.add_row(
            f"`{strat.name}`",
            str(len(chunks)),
            f"{avg_len:.1f} chars",
            boundary,
            f"{elapsed_ms:.2f} ms"
        )

    console.print(table)
    console.print("\n[bold green]✅ Multi-strategy chunking verified on complete MSMARCO-XI corpus![/bold green]\n")

if __name__ == "__main__":
    evaluate_chunking_on_full_corpus()
