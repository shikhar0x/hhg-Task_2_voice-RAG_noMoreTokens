import pytest
from chunking.strategies import (
    FixedWindowChunker,
    RecursiveSentenceChunker,
    SemanticParagraphChunker,
    get_chunker
)

def test_fixed_window_chunker(sample_short_text):
    chunker = FixedWindowChunker(chunk_size=30, overlap=10)
    chunks = chunker.chunk(sample_short_text)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert c["metadata"]["strategy"] == "fixed_window"
        assert len(c["text"]) <= 30

def test_recursive_sentence_chunker(sample_multi_sentence_text):
    chunker = RecursiveSentenceChunker(max_words=15)
    chunks = chunker.chunk(sample_multi_sentence_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert "text" in c
        assert c["metadata"]["strategy"] == "recursive_sentence"

def test_semantic_paragraph_chunker(sample_paragraph_text):
    chunker = SemanticParagraphChunker(max_length=500)
    chunks = chunker.chunk(sample_paragraph_text)
    assert len(chunks) == 2
    assert "First paragraph" in chunks[0]["text"]
    assert "Second paragraph" in chunks[1]["text"]

def test_get_chunker_factory():
    fw = get_chunker("fixed_window")
    assert isinstance(fw, FixedWindowChunker)

    rs = get_chunker("recursive_sentence")
    assert isinstance(rs, RecursiveSentenceChunker)

    sp = get_chunker("semantic_paragraph")
    assert isinstance(sp, SemanticParagraphChunker)

    fallback = get_chunker("unknown_strategy")
    assert isinstance(fallback, RecursiveSentenceChunker)
