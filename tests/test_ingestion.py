import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.document_parser import parse_txt, parse_csv, parse_markdown
from src.ingestion.chunker import RecursiveChunker


def test_parse_txt_returns_content():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Hello world. This is a test document.")
        path = f.name
    try:
        docs = parse_txt(path)
        assert len(docs) == 1
        assert "Hello world" in docs[0]["content"]
        assert docs[0]["source"] == os.path.basename(path)
    finally:
        os.unlink(path)


def test_parse_txt_metadata():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Data pipeline content.")
        path = f.name
    try:
        docs = parse_txt(path)
        assert docs[0]["metadata"]["type"] == "txt"
    finally:
        os.unlink(path)


def test_parse_csv_each_row_is_doc():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8", newline="") as f:
        f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        path = f.name
    try:
        docs = parse_csv(path)
        assert len(docs) == 2
        assert "Alice" in docs[0]["content"]
        assert "Bob" in docs[1]["content"]
    finally:
        os.unlink(path)


def test_chunker_splits_long_text():
    chunker = RecursiveChunker(chunk_size=100, overlap=10)
    text = "word " * 100  # 500 chars
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 150  # some tolerance for overlap


def test_chunker_short_text_single_chunk():
    chunker = RecursiveChunker(chunk_size=512, overlap=64)
    text = "Short text."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunker_split_documents_returns_document_chunks():
    chunker = RecursiveChunker(chunk_size=200, overlap=20)
    docs = [{"source": "test.txt", "content": "sentence one. " * 30, "metadata": {"type": "txt"}}]
    chunks = chunker.split_documents(docs)
    assert len(chunks) > 0
    from src.transformation.schema import DocumentChunk
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)


def test_chunker_chunk_indices_sequential():
    chunker = RecursiveChunker(chunk_size=100, overlap=10)
    docs = [{"source": "f.txt", "content": "word " * 100, "metadata": {}}]
    chunks = chunker.split_documents(docs)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
