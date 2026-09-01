import time
from typing import List
from sentence_transformers import SentenceTransformer
from src.transformation.schema import DocumentChunk

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return vectors.tolist()

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> tuple:
        """Returns (chunks_with_embeddings, elapsed_seconds)."""
        start = time.time()
        texts = [c.content for c in chunks]
        vectors = self.embed_texts(texts, batch_size=batch_size)
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
        elapsed = time.time() - start
        return chunks, elapsed

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
