from pathlib import Path
from typing import List, Tuple
import chromadb
from src.transformation.schema import DocumentChunk


class ChromaStore:
    """ChromaDB persistent store — alternative to FAISS."""

    def __init__(self, persist_dir: str = "data/processed/chroma"):
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection("documents")

    def add(self, chunks: List[DocumentChunk]) -> None:
        ids = [c.chunk_id for c in chunks]
        docs = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index,
                      "token_count": c.token_count, **c.metadata} for c in chunks]
        self._col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        results = self._col.query(query_embeddings=[query_embedding], n_results=top_k)
        output = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunk = DocumentChunk(
                chunk_id=cid,
                source=meta["source"],
                content=doc,
                chunk_index=meta["chunk_index"],
                token_count=meta["token_count"],
                char_count=len(doc),
                metadata={k: v for k, v in meta.items() if k not in ("source", "chunk_index", "token_count")},
            )
            output.append((chunk, float(dist)))
        return output
