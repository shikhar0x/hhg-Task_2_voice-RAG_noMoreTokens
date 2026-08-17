import time
import numpy as np
import re
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult

_documents = []
_metadatas = []
_vocab = {}
_idf = None
_doc_matrix = None

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "whatever", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def log_numpy_and_corpus_diagnostics():
    """Prints NumPy BLAS/LAPACK configuration diagnostic."""
    try:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            np.show_config()
        config_str = f.getvalue().strip()
        logger.debug(f"NumPy System Configuration:\n{config_str}")
    except Exception as e:
        logger.warning(f"Could not retrieve numpy configuration: {e}")

def build_fast_vector_index(docs: list[str], metadatas: list[dict]):
    """Builds and pre-warms the in-memory TF-IDF cosine vector index."""
    global _documents, _metadatas, _vocab, _idf, _doc_matrix
    _documents = docs
    _metadatas = metadatas

    vocab = {}
    doc_word_counts = []
    doc_freq = {}
    
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        counts = {}
        seen_in_doc = set()
        for w in words:
            counts[w] = counts.get(w, 0) + 1
            if w not in vocab:
                vocab[w] = len(vocab)
            if w not in seen_in_doc:
                doc_freq[w] = doc_freq.get(w, 0) + 1
                seen_in_doc.add(w)
        doc_word_counts.append(counts)

    _vocab = vocab
    vocab_size = len(vocab)
    doc_count = len(docs)

    # Compute Inverse Document Frequency (IDF) with stopword penalty
    idf = np.zeros(vocab_size, dtype=np.float32)
    for w, idx in vocab.items():
        df = doc_freq.get(w, 1)
        if w in ENGLISH_STOPWORDS:
            idf[idx] = 0.05  # Heavily suppress stopword contribution
        else:
            idf[idx] = np.log((doc_count + 1.0) / (df + 1.0)) + 1.0

    _idf = idf

    matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)
    for i, counts in enumerate(doc_word_counts):
        for w, count in counts.items():
            idx = vocab[w]
            matrix[i, idx] = (1.0 + np.log(count)) * idf[idx]
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm

    _doc_matrix = matrix
    log_numpy_and_corpus_diagnostics()
    logger.debug(f"Pre-warmed in-memory TF-IDF vector index ({doc_count} passages, {vocab_size} vocab dimensions). Matrix shape={matrix.shape}, dtype={matrix.dtype}.")

def warmup_vector_index():
    """Initializes the fast vector index at startup to eliminate query-1 cold start."""
    global _doc_matrix
    if _doc_matrix is None:
        client = _get_chroma_client()
        col = client.get_or_create_collection("msmarco_corpus")
        data = col.get()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        if docs:
            build_fast_vector_index(docs, metas)

_chroma_client = None


def _get_chroma_client():
    """Single shared ChromaDB client to avoid multi-client 'database is locked' errors."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    return _chroma_client


def get_vector_store():
    return _get_chroma_client().get_or_create_collection("msmarco_corpus")

class VectorRetrievalStep(BaseStep):
    """Sub-5ms TF-IDF Vector Retrieval Step."""
    name = "retrieval_vector"

    def __init__(self):
        warmup_vector_index()

    def execute(self, input_data: dict) -> StepResult:
        global _documents, _metadatas, _vocab, _idf, _doc_matrix
        query = input_data.get("transcript", "").strip()
        top_k = input_data.get("top_k", 3)

        if not query:
            return StepResult(success=False, error="Query text is empty.")

        if _doc_matrix is None:
            warmup_vector_index()

        # Vectorize query using TF-IDF
        q_words = re.findall(r'\w+', query.lower())
        q_counts = {}
        for w in q_words:
            q_counts[w] = q_counts.get(w, 0) + 1

        q_vec = np.zeros(len(_vocab), dtype=np.float32)
        has_non_stopword = False
        for w, count in q_counts.items():
            if w in _vocab:
                idx = _vocab[w]
                q_vec[idx] = (1.0 + np.log(count)) * _idf[idx]
                if w not in ENGLISH_STOPWORDS:
                    has_non_stopword = True

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0 and has_non_stopword:
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
