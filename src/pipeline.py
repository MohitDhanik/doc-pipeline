import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.ingestion.document_parser import parse_files
from src.ingestion.chunker import RecursiveChunker
from src.transformation.quality_checks import QualityChecker
from src.transformation.schema import QualityReport
from src.transformation.embedder import Embedder
from src.store.faiss_store import FAISSStore

DEFAULT_STORE_PATH = "data/processed/faiss"


@dataclass
class PipelineResult:
    n_docs: int
    n_chunks_raw: int
    n_chunks_stored: int
    quality_report: QualityReport
    stages_timing: dict
    store_path: str
    embed_time: float


class Pipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chars: int = 50,
        store_path: str = DEFAULT_STORE_PATH,
        embedder: Optional[Embedder] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars
        self.store_path = store_path
        self.embedder = embedder or Embedder()
        self._store: Optional[FAISSStore] = None

    def run(self, file_paths: List[str]) -> PipelineResult:
        timings = {}

        # Stage 1: Parse
        t0 = time.time()
        docs = parse_files(file_paths)
        timings["parse"] = time.time() - t0

        # Stage 2: Chunk
        t0 = time.time()
        chunker = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        chunks = chunker.split_documents(docs)
        n_raw = len(chunks)
        timings["chunk"] = time.time() - t0

        # Stage 3: Quality checks
        t0 = time.time()
        checker = QualityChecker()
        report = checker.run_all(chunks)
        good_chunks = checker.filter_passing(chunks, report)
        timings["quality"] = time.time() - t0

        # Stage 4: Embed
        t0 = time.time()
        good_chunks, embed_time = self.embedder.embed_chunks(good_chunks)
        timings["embed"] = time.time() - t0

        # Stage 5: Store
        t0 = time.time()
        dim = len(good_chunks[0].embedding) if good_chunks else 384
        store = FAISSStore(dim=dim)
        store.add(good_chunks)
        store.save(self.store_path)
        self._store = store
        timings["store"] = time.time() - t0

        return PipelineResult(
            n_docs=len(docs),
            n_chunks_raw=n_raw,
            n_chunks_stored=len(good_chunks),
            quality_report=report,
            stages_timing=timings,
            store_path=self.store_path,
            embed_time=embed_time,
        )

    def load_store(self) -> FAISSStore:
        if self._store is None:
            self._store = FAISSStore.load(self.store_path)
        return self._store


if __name__ == "__main__":
    import glob
    raw_files = glob.glob("data/raw/*.txt") + glob.glob("data/raw/*.md") + glob.glob("data/raw/*.csv")
    print(f"Running pipeline on {len(raw_files)} files...")
    pipeline = Pipeline()
    result = pipeline.run(raw_files)
    print(f"Docs parsed     : {result.n_docs}")
    print(f"Chunks raw      : {result.n_chunks_raw}")
    print(f"Chunks stored   : {result.n_chunks_stored}")
    print(f"Quality pass    : {result.quality_report.pass_rate:.1%}")
    print(f"Embed time      : {result.embed_time:.2f}s")
    print(f"Store saved to  : {result.store_path}")
    print("Stage timings   :")
    for stage, t in result.stages_timing.items():
        print(f"  {stage:10s}: {t:.3f}s")
