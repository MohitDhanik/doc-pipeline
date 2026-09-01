import json
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from src.transformation.schema import DocumentChunk


class FAISSStore:
    """Flat L2 index wrapping FAISS. Stores chunk metadata alongside index."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self._chunks: List[DocumentChunk] = []

    def add(self, chunks: List[DocumentChunk]) -> None:
        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {vectors.shape}")
        self.index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self._chunks:
            return []
        top_k = min(top_k, len(self._chunks))
        q = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((self._chunks[idx], float(dist)))
        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        # Save chunks without embedding arrays to keep file small
        slim = []
        for c in self._chunks:
            d = {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "char_count": c.char_count,
                "metadata": c.metadata,
            }
            slim.append(d)
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FAISSStore":
        path = Path(directory)
        index = faiss.read_index(str(path / "index.faiss"))
        dim = index.d
        store = cls(dim=dim)
        store.index = index
        with open(path / "chunks.json", encoding="utf-8") as f:
            raw = json.load(f)
        from src.transformation.schema import DocumentChunk
        for d in raw:
            store._chunks.append(DocumentChunk(**d))
        return store

    def __len__(self):
        return len(self._chunks)
