"""Tiny eval harness: retrieval hit-rate and answer keyword coverage."""

from __future__ import annotations

import json
from pathlib import Path

from rag.answer import answer
from rag.retriever import retrieve

QUESTIONS = Path(__file__).parent / "questions.json"


def hit_rate(retrieved: list[dict], expected_doc: str) -> int:
    return int(any(c["doc_id"] == expected_doc for c in retrieved))


def keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text_lc = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lc)
    return found / len(keywords)


def main() -> None:
    items = json.loads(QUESTIONS.read_text())
    rows = []
    hits = 0
    coverage_sum = 0.0
    for item in items:
        q = item["question"]
        expected_doc = item.get("expected_doc_id", "")
        keywords = item.get("expected_keywords", [])
        retrieved = retrieve(q)
        h = hit_rate(retrieved, expected_doc) if expected_doc else None
        result = answer(q)
        kc = keyword_coverage(result["answer"], keywords)
        rows.append({"q": q, "hit": h, "kw_coverage": round(kc, 2)})
        if h is not None:
            hits += h
        coverage_sum += kc

    n = len(items)
    print(f"\n{'Question':<60} {'hit@k':<7} {'kw cov':<7}")
    print("-" * 75)
    for r in rows:
        q = (r["q"][:55] + "…") if len(r["q"]) > 56 else r["q"]
        hit = "—" if r["hit"] is None else str(r["hit"])
        print(f"{q:<60} {hit:<7} {r['kw_coverage']:<7}")
    print("-" * 75)
    if n:
        print(f"Hit-rate@k: {hits}/{n} = {hits/n:.1%}")
        print(f"Mean keyword coverage: {coverage_sum/n:.1%}")


if __name__ == "__main__":
    main()
