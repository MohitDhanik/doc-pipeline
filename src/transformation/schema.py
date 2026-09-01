from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class DocumentChunk:
    chunk_id: str        # uuid4 string
    source: str          # originating filename
    content: str         # text content of this chunk
    chunk_index: int     # position within source document
    token_count: int     # approximate word-based token count
    char_count: int      # character count
    embedding: Optional[List[float]] = None   # dense vector after embedding stage
    metadata: Dict = field(default_factory=dict)  # arbitrary extra fields (page, row, etc.)


@dataclass
class QualityReport:
    total: int
    pass_count: int
    fail_count: int
    pass_rate: float
    issues: List[Dict]   # each: {chunk_id, reason}
