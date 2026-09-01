import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transformation.schema import DocumentChunk
from src.transformation.quality_checks import QualityChecker


def make_chunk(content, chunk_id=None, source="test.txt"):
    import uuid
    return DocumentChunk(
        chunk_id=chunk_id or str(uuid.uuid4()),
        source=source,
        content=content,
        chunk_index=0,
        token_count=len(content.split()),
        char_count=len(content),
    )


def test_check_null_content_flags_empty():
    checker = QualityChecker()
    chunks = [make_chunk(""), make_chunk("   "), make_chunk("valid content here")]
    issues = checker.check_null_content(chunks)
    assert len(issues) == 2


def test_check_min_length_flags_short():
    checker = QualityChecker()
    chunks = [make_chunk("hi"), make_chunk("x" * 51)]
    issues = checker.check_min_length(chunks, min_chars=50)
    assert len(issues) == 1
    assert "below_min_length" in issues[0]["reason"]


def test_check_duplicates_flags_exact_copy():
    checker = QualityChecker()
    c1 = make_chunk("duplicate content here", chunk_id="aaa")
    c2 = make_chunk("duplicate content here", chunk_id="bbb")
    c3 = make_chunk("unique content xyz", chunk_id="ccc")
    issues = checker.check_duplicates([c1, c2, c3])
    assert len(issues) == 1
    assert issues[0]["chunk_id"] == "bbb"


def test_run_all_returns_quality_report():
    checker = QualityChecker()
    chunks = [make_chunk("good content " * 5) for _ in range(5)]
    report = checker.run_all(chunks)
    assert report.total == 5
    assert report.pass_count + report.fail_count == report.total
    assert 0.0 <= report.pass_rate <= 1.0


def test_filter_passing_removes_bad_chunks():
    checker = QualityChecker()
    good = make_chunk("this is sufficiently long content for the check " * 2)
    bad = make_chunk("")
    report = checker.run_all([good, bad])
    passing = checker.filter_passing([good, bad], report)
    assert len(passing) == 1
    assert passing[0].chunk_id == good.chunk_id


def test_quality_pass_rate_all_good():
    checker = QualityChecker()
    # Use distinct content so dedup check doesn't flag them
    chunks = [make_chunk(f"valid unique content number {i} " * 5) for i in range(10)]
    report = checker.run_all(chunks)
    assert report.pass_rate == 1.0
