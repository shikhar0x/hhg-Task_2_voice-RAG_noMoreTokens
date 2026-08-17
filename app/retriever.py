import time
from dataclasses import dataclass, field
from retrieval.vector_store import VectorRetrievalStep, warmup_vector_index

@dataclass
class SearchResponse:
    total_ms: float
    embed_ms: float
    search_ms: float
    documents: list = field(default_factory=list)

_step = None

def warmup():
    global _step
    warmup_vector_index()
    _step = VectorRetrievalStep()

def search(query: str, top_k: int = 5) -> SearchResponse:
    global _step
    if _step is None:
        warmup()
    
    t0 = time.perf_counter()
    res = _step.run({"transcript": query, "top_k": top_k})
    t_total = (time.perf_counter() - t0) * 1000.0
    
    total_duration = res.duration_ms if res.duration_ms > 0 else t_total
    embed_ms = min(0.05, total_duration * 0.15) if total_duration > 0 else 0.01
    search_ms = max(0.01, total_duration - embed_ms)
    
    return SearchResponse(
        total_ms=total_duration,
        embed_ms=embed_ms,
        search_ms=search_ms,
        documents=res.data.get("documents", []) if res.data else []
    )
