"""End-to-end RAG: retrieve chunks, ask the LLM, return grounded answer + sources."""

from __future__ import annotations

from rag.llm import chat
from rag.retriever import retrieve

SYSTEM_PROMPT = (
    "You are a careful technical assistant for automotive engineering papers. "
    "Answer ONLY using the provided context. If the context does not contain "
    "the answer, say so explicitly. Cite sources inline as [doc:page]. "
    "Be concise and precise."
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['doc_id']}:p{c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question: str, k: int | None = None, temperature: float = 0.0) -> dict:
    chunks = retrieve(question, k=k)
    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    reply = chat(messages, temperature=temperature)
    return {"answer": reply, "sources": chunks}
