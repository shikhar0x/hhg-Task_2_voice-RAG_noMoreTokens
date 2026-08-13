from abc import ABC, abstractmethod
import re
from typing import Any

class BaseChunker(ABC):
    name: str = "base_chunker"

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        pass

class FixedWindowChunker(BaseChunker):
    """Strategy 1: Sliding token/char window with configurable overlap."""
    name = "fixed_window"

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        meta = metadata or {}
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**meta, "strategy": self.name, "start": start, "end": end}
                })
            if end >= text_len:
                break
            start += max(1, self.chunk_size - self.overlap)
        return chunks

class RecursiveSentenceChunker(BaseChunker):
    """Strategy 2: Boundary-aware sentence splitter preserving syntactic cohesion."""
    name = "recursive_sentence"

    def __init__(self, max_words: int = 80):
        self.max_words = max_words

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        meta = metadata or {}
        # Split by sentence terminators
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        chunks = []
        current_chunk: list[str] = []
        current_words = 0

        for sentence in sentences:
            s_words = len(sentence.split())
            if current_words + s_words > self.max_words and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": {**meta, "strategy": self.name}
                })
                current_chunk = []
                current_words = 0
            current_chunk.append(sentence)
            current_words += s_words

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": {**meta, "strategy": self.name}
            })
        return chunks

class SemanticParagraphChunker(BaseChunker):
    """Strategy 3: Structural paragraph & header-aware chunker."""
    name = "semantic_paragraph"

    def __init__(self, max_length: int = 600):
        self.max_length = max_length

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        meta = metadata or {}
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        for p in paragraphs:
            if len(p) <= self.max_length:
                chunks.append({"text": p, "metadata": {**meta, "strategy": self.name}})
            else:
                sub_chunker = RecursiveSentenceChunker(max_words=80)
                chunks.extend(sub_chunker.chunk(p, metadata=meta))
        return chunks

def get_chunker(strategy_name: str) -> BaseChunker:
    strategies = {
        "fixed_window": FixedWindowChunker(),
        "recursive_sentence": RecursiveSentenceChunker(),
        "semantic_paragraph": SemanticParagraphChunker()
    }
    return strategies.get(strategy_name, RecursiveSentenceChunker())
