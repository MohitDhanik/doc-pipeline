import uuid
from typing import List, Dict
from src.transformation.schema import DocumentChunk


class RecursiveChunker:
    """
    Splits text recursively on paragraph boundaries first, then sentences,
    then words — ensuring chunks stay near chunk_size with overlap stitching.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            # character-level fallback
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # part itself may be too long — recurse with next separator
                if len(part) > self.chunk_size and next_seps:
                    sub = self._split_text(part, next_seps)
                    # add overlap: prepend tail of previous chunk to first sub
                    if chunks and sub:
                        overlap_text = chunks[-1][-self.overlap:]
                        sub[0] = (overlap_text + " " + sub[0]).strip()
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part.strip()

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_text(text, self.SEPARATORS)

    def split_documents(self, docs: List[Dict]) -> List[DocumentChunk]:
        chunks = []
        for doc in docs:
            texts = self.split_text(doc["content"])
            for i, text in enumerate(texts):
                token_count = len(text.split())
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    source=doc["source"],
                    content=text,
                    chunk_index=i,
                    token_count=token_count,
                    char_count=len(text),
                    metadata=doc.get("metadata", {}),
                ))
        return chunks
