"""Retrieve top-k chunks from the FAISS index for a query."""

from __future__ import annotations

import pickle
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import INDEX_DIR, settings


@lru_cache(maxsize=1)
def _load():
    if not (INDEX_DIR / "faiss.index").exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_DIR}. Run `python -m rag.ingest` first."
        )
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(settings.embedding_model)
    return index, chunks, model


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or settings.top_k
    index, chunks, model = _load()
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q, k)
    out = []
    for score, i in zip(scores[0].tolist(), idx[0].tolist()):
        if i < 0:
            continue
        c = chunks[i]
        out.append({**c, "score": float(score)})
    return out
