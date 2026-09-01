import csv
import io
from pathlib import Path
from typing import List, Dict


def parse_txt(file_path: str) -> List[Dict]:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    return [{"source": path.name, "content": content, "metadata": {"type": "txt"}}]


def parse_markdown(file_path: str) -> List[Dict]:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    # Strip markdown syntax for plain text processing
    import re
    content = re.sub(r"#{1,6}\s+", "", content)
    content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
    content = re.sub(r"\*(.+?)\*", r"\1", content)
    content = re.sub(r"`{1,3}[^`]*`{1,3}", "", content)
    return [{"source": path.name, "content": content, "metadata": {"type": "markdown"}}]


def parse_pdf(file_path: str) -> List[Dict]:
    import pdfplumber
    path = Path(file_path)
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append({
                    "source": path.name,
                    "content": text,
                    "metadata": {"type": "pdf", "page": i + 1},
                })
    return docs


def parse_csv(file_path: str) -> List[Dict]:
    path = Path(file_path)
    docs = []
    with open(file_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            if content.strip():
                docs.append({
                    "source": path.name,
                    "content": content,
                    "metadata": {"type": "csv", "row": i + 1},
                })
    return docs


def parse_file(file_path: str) -> List[Dict]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        return parse_markdown(file_path)
    elif suffix == ".csv":
        return parse_csv(file_path)
    else:
        return parse_txt(file_path)


def parse_files(file_paths: List[str]) -> List[Dict]:
    docs = []
    for fp in file_paths:
        docs.extend(parse_file(fp))
    return docs
