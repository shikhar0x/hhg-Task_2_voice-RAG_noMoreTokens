import chromadb
from chromadb.utils import embedding_functions
from config.settings import settings
from config.logger import logger
from harness.base import BaseStep, StepResult

_client = None
_collection = None

def get_vector_store(collection_name: str = "msmarco_corpus"):
    global _client, _collection
    if _collection is None:
        logger.debug(f"Connecting to ChromaDB at {settings.chroma_path}")
        _client = chromadb.PersistentClient(path=settings.chroma_path)
        emb_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = _client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

class VectorRetrievalStep(BaseStep):
    """Retrieval step querying vector DB with cosine similarity extraction."""
    name = "retrieval"

    def __init__(self, collection_name: str = "msmarco_corpus"):
        self.collection_name = collection_name
        self.col = get_vector_store(collection_name)

    def execute(self, input_data: dict) -> StepResult:
        query = input_data.get("transcript", "").strip()
        top_k = input_data.get("top_k", settings.default_top_k)

        if not query:
            return StepResult(success=False, error="Query text is empty.")

        results = self.col.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        docs = results["documents"][0] if results.get("documents") else []
        distances = results["distances"][0] if results.get("distances") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []

        # Convert cosine distance to cosine similarity (1.0 - distance)
        similarities = [max(0.0, 1.0 - float(d)) for d in distances]
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
