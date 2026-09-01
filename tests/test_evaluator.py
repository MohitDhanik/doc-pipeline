import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.evaluator import precision_at_k, recall_at_k


def test_precision_at_k_all_relevant():
    retrieved = ["a.txt", "b.txt", "c.txt"]
    relevant = ["a.txt", "b.txt", "c.txt"]
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = ["x.txt", "y.txt"]
    relevant = ["a.txt"]
    assert precision_at_k(retrieved, relevant, k=2) == 0.0


def test_precision_at_k_partial():
    retrieved = ["a.txt", "x.txt", "b.txt", "y.txt"]
    relevant = ["a.txt", "b.txt"]
    # k=2: only a.txt matches → 1/2
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    # k=4: a.txt and b.txt match → 2/4
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_precision_at_k_zero():
    assert precision_at_k([], ["a.txt"], k=0) == 0.0


def test_recall_at_k_all_found():
    retrieved = ["a.txt", "b.txt"]
    relevant = ["a.txt", "b.txt"]
    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_recall_at_k_partial():
    retrieved = ["a.txt", "x.txt", "b.txt"]
    relevant = ["a.txt", "b.txt", "c.txt"]
    # k=3: found a.txt, b.txt → 2/3
    assert abs(recall_at_k(retrieved, relevant, k=3) - 2/3) < 1e-9


def test_recall_at_k_none_found():
    retrieved = ["x.txt", "y.txt"]
    relevant = ["a.txt", "b.txt"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a.txt"], [], k=1) == 0.0


def test_precision_only_uses_top_k():
    retrieved = ["x.txt", "x.txt", "a.txt"]  # a.txt is at index 2
    relevant = ["a.txt"]
    # k=2: x.txt, x.txt — no match
    assert precision_at_k(retrieved, relevant, k=2) == 0.0
    # k=3: includes a.txt — 1 match → 1/3
    assert abs(precision_at_k(retrieved, relevant, k=3) - 1/3) < 1e-9
