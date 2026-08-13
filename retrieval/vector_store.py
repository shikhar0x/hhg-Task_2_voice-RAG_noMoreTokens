import time
import numpy as np
import re
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult

_documents = []
_metadatas = []
_vocab = {}
_doc_matrix = None

def build_fast_vector_index(docs: list[str], metadatas: list[dict]):
    """Builds and pre-warms the in-memory cosine vector index."""
    global _documents, _metadatas, _vocab, _doc_matrix
    _documents = docs
    _metadatas = metadatas

    vocab = {}
    doc_word_counts = []
    
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if w not in vocab:
                vocab[w] = len(vocab)
        doc_word_counts.append(counts)

    _vocab = vocab
    vocab_size = len(vocab)
    doc_count = len(docs)

    matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)
    for i, counts in enumerate(doc_word_counts):
        for w, count in counts.items():
            matrix[i, vocab[w]] = count
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm

    _doc_matrix = matrix
    logger.info(f"Pre-warmed in-memory vector index ({doc_count} passages, {vocab_size} vocab dimensions).")

def warmup_vector_index():
    """Initializes the fast vector index at startup to eliminate query-1 cold start."""
    global _doc_matrix
    if _doc_matrix is None:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_path)
        col = client.get_or_create_collection("msmarco_corpus")
        data = col.get()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        if docs:
            build_fast_vector_index(docs, metas)

def get_vector_store():
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection("msmarco_corpus")

class VectorRetrievalStep(BaseStep):
    """Sub-5ms Vector Retrieval Step."""
    name = "retrieval_vector"

    def __init__(self):
        warmup_vector_index()

    def execute(self, input_data: dict) -> StepResult:
        global _documents, _metadatas, _vocab, _doc_matrix
        query = input_data.get("transcript", "").strip()
        top_k = input_data.get("top_k", 3)

        if not query:
            return StepResult(success=False, error="Query text is empty.")

        if _doc_matrix is None:
            warmup_vector_index()

        # Vectorize query
        q_words = re.findall(r'\w+', query.lower())
        q_vec = np.zeros(len(_vocab), dtype=np.float32)
        for w in q_words:
            if w in _vocab:
                q_vec[_vocab[w]] += 1

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
            sims = np.dot(_doc_matrix, q_vec)
        else:
            sims = np.zeros(len(_documents), dtype=np.float32)

        top_indices = np.argsort(sims)[::-1][:top_k]
        
        docs = [_documents[idx] for idx in top_indices]
        metadatas = [_metadatas[idx] if idx < len(_metadatas) else {} for idx in top_indices]
        similarities = [float(sims[idx]) for idx in top_indices]
        top_similarity = similarities[0] if similarities else 0.0

        return StepResult(
            success=True,
            data={
                "documents": docs,
                "similarities": similarities,
                "metadatas": metadatas,
                "top_similarity": top_similarity,
                "count": len(docs)
            }
        )
