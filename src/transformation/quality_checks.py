import hashlib
from typing import List
from src.transformation.schema import DocumentChunk, QualityReport


class QualityChecker:

    def check_null_content(self, chunks: List[DocumentChunk]) -> List[dict]:
        issues = []
        for c in chunks:
            if not c.content or not c.content.strip():
                issues.append({"chunk_id": c.chunk_id, "source": c.source, "reason": "null_or_empty_content"})
        return issues

    def check_min_length(self, chunks: List[DocumentChunk], min_chars: int = 50) -> List[dict]:
        issues = []
        for c in chunks:
            if c.char_count < min_chars:
                issues.append({"chunk_id": c.chunk_id, "source": c.source,
                                "reason": f"below_min_length({c.char_count}<{min_chars})"})
        return issues

    def check_duplicates(self, chunks: List[DocumentChunk]) -> List[dict]:
        seen = {}
        issues = []
        for c in chunks:
            h = hashlib.md5(c.content.strip().encode()).hexdigest()
            if h in seen:
                issues.append({"chunk_id": c.chunk_id, "source": c.source,
                                "reason": f"duplicate_of:{seen[h]}"})
            else:
                seen[h] = c.chunk_id
        return issues

    def run_all(self, chunks: List[DocumentChunk]) -> QualityReport:
        issues = []
        issues.extend(self.check_null_content(chunks))
        issues.extend(self.check_min_length(chunks))
        issues.extend(self.check_duplicates(chunks))

        failed_ids = {i["chunk_id"] for i in issues}
        fail_count = len(failed_ids)
        pass_count = len(chunks) - fail_count
        pass_rate = pass_count / len(chunks) if chunks else 0.0

        return QualityReport(
            total=len(chunks),
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            issues=issues,
        )

    def filter_passing(self, chunks: List[DocumentChunk], report: QualityReport) -> List[DocumentChunk]:
        failed_ids = {i["chunk_id"] for i in report.issues}
        return [c for c in chunks if c.chunk_id not in failed_ids]
