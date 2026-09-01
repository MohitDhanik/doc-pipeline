from dataclasses import dataclass
from typing import List
from src.transformation.schema import DocumentChunk
from src.transformation.embedder import Embedder
from src.store.faiss_store import FAISSStore


@dataclass
class RetrievalResult:
    rank: int
    chunk: DocumentChunk
    score: float          # L2 distance (lower = more similar)
    similarity: float     # converted cosine-like score in [0,1]


def _l2_to_similarity(dist: float) -> float:
    """Convert L2 distance to a 0-1 similarity score."""
    return 1.0 / (1.0 + dist)


class Retriever:
    def __init__(self, store: FAISSStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.embedder.embed_query(text)
        raw = self.store.search(q_emb, top_k=top_k)
        results = []
        for rank, (chunk, dist) in enumerate(raw, start=1):
            results.append(RetrievalResult(
                rank=rank,
                chunk=chunk,
                score=dist,
                similarity=_l2_to_similarity(dist),
            ))
        return results
