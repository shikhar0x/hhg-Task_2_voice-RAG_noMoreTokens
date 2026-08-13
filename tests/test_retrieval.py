import pytest
from retrieval.vector_store import VectorRetrievalStep, build_fast_vector_index

def test_vector_retrieval_execute_in_memory(sample_corpus):
    docs, metadatas = sample_corpus
    # Build in-memory test index
    build_fast_vector_index(docs, metadatas)

    step = VectorRetrievalStep()
    res = step.run({"transcript": "what is a corporation?", "top_k": 2})

    assert res.success is True
    data = res.data
    assert len(data["documents"]) == 2
    assert "corporation" in data["documents"][0].lower()
    assert data["top_similarity"] > 0.0
    assert res.duration_ms > 0.0

def test_vector_retrieval_empty_query():
    step = VectorRetrievalStep()
    res = step.run({"transcript": ""})
    assert res.success is False
    assert "empty" in res.error.lower()
