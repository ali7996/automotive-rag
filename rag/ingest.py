"""Build a FAISS index from PDFs in data/."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pymupdf4llm
from sentence_transformers import SentenceTransformer

from rag.config import DATA_DIR, INDEX_DIR, settings


@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str


def _split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def _is_low_quality(text: str) -> bool:
    """Skip chunks that are mostly numbers, single chars per line, or empty after cleanup."""
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    alpha = sum(c.isalpha() for c in stripped)
    if alpha / max(len(stripped), 1) < 0.4:
        return True
    return False


def _read_pdf(path: Path) -> list[Chunk]:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    chunks: list[Chunk] = []
    for entry in pages:
        page_num = entry.get("metadata", {}).get("page", 0) + 1
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        for piece in _split(text, settings.chunk_size, settings.chunk_overlap):
            if _is_low_quality(piece):
                continue
            chunks.append(Chunk(doc_id=path.name, page=page_num, text=piece))
    return chunks


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> int:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}. Add automotive papers first.")

    print(f"Found {len(pdfs)} PDF(s).")
    chunks: list[Chunk] = []
    for pdf in pdfs:
        print(f"  reading {pdf.name}")
        chunks.extend(_read_pdf(pdf))
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model {settings.embedding_model} (first run downloads ~130MB)…")
    model = SentenceTransformer(settings.embedding_model)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump([asdict(c) for c in chunks], f)

    meta = {
        "num_chunks": len(chunks),
        "num_docs": len(pdfs),
        "docs": [p.name for p in pdfs],
        "embedding_model": settings.embedding_model,
        "dim": int(embeddings.shape[1]),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Index written to {index_dir}")
    return len(chunks)


if __name__ == "__main__":
    build_index()
