import os
import sys
import uuid
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transformation.schema import DocumentChunk
from src.store.faiss_store import FAISSStore


DIM = 8  # small dim for fast tests


def make_chunk_with_embedding(content="test content", dim=DIM):
    vec = np.random.randn(dim).astype(np.float32).tolist()
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        source="test.txt",
        content=content,
        chunk_index=0,
        token_count=len(content.split()),
        char_count=len(content),
        embedding=vec,
    )


def test_faiss_store_add_and_search_returns_correct_count():
    store = FAISSStore(dim=DIM)
    chunks = [make_chunk_with_embedding(f"content {i}") for i in range(10)]
    store.add(chunks)
    q = np.random.randn(DIM).astype(np.float32).tolist()
    results = store.search(q, top_k=5)
    assert len(results) == 5


def test_faiss_store_search_returns_tuples():
    store = FAISSStore(dim=DIM)
    chunks = [make_chunk_with_embedding()]
    store.add(chunks)
    q = np.random.randn(DIM).astype(np.float32).tolist()
    results = store.search(q, top_k=1)
    assert len(results) == 1
    chunk, score = results[0]
    assert isinstance(chunk, DocumentChunk)
    assert isinstance(score, float)


def test_faiss_store_search_empty_returns_empty():
    store = FAISSStore(dim=DIM)
    q = np.random.randn(DIM).astype(np.float32).tolist()
    results = store.search(q, top_k=5)
    assert results == []


def test_faiss_store_top_k_clamps_to_available():
    store = FAISSStore(dim=DIM)
    chunks = [make_chunk_with_embedding(f"doc {i}") for i in range(3)]
    store.add(chunks)
    q = np.random.randn(DIM).astype(np.float32).tolist()
    results = store.search(q, top_k=10)
    assert len(results) == 3  # only 3 available


def test_faiss_store_save_and_load(tmp_path):
    store = FAISSStore(dim=DIM)
    chunks = [make_chunk_with_embedding(f"saved chunk {i}") for i in range(5)]
    store.add(chunks)
    save_dir = str(tmp_path / "faiss_test")
    store.save(save_dir)

    loaded = FAISSStore.load(save_dir)
    assert len(loaded) == 5
    q = np.random.randn(DIM).astype(np.float32).tolist()
    results = loaded.search(q, top_k=3)
    assert len(results) == 3


def test_faiss_store_len():
    store = FAISSStore(dim=DIM)
    assert len(store) == 0
    chunks = [make_chunk_with_embedding() for _ in range(7)]
    store.add(chunks)
    assert len(store) == 7
