---
title: Automotive RAG
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🚗 Automotive RAG

Question-answering over automotive engineering papers. Retrieves relevant passages, grounds an LLM in them, and cites sources inline.

Built to be **provider-agnostic**: the same code runs against a hosted LLM (Groq, free tier) for the public demo, or against a fully local model (Ollama) for offline use.

> **Live demo:** _coming soon — Hugging Face Space link goes here once deployed_
> **Author:** [@ali7996](https://github.com/ali7996)

---

## Why this project

Most public RAG demos pick one backend (an API key OR a local model) and hardcode it. Real-world deployments need both — privacy-sensitive environments demand local inference, while public-facing demos need hosted speed. This project shows that the same codebase can serve both, configured by a single environment variable.

The domain is automotive papers because that's where the targeted research roles sit (Mercedes-Benz EDS.AI, VW DataLab, etc.) — but the pipeline is domain-agnostic. Drop any PDFs in `data/` and re-index.

## Architecture

```
                ┌─ Gradio UI (app.py)
                │
        question│
                ▼
    ┌──────────────────────┐      top-k chunks       ┌──────────────────┐
    │ Retriever            │◀────────────────────────│ FAISS (in-memory)│
    │ bge-small-en-v1.5    │                         │ chunks + meta    │
    └──────────────────────┘                         └──────────────────┘
                │
                │ (context + question)
                ▼
        ┌───────────────┐                ┌─────────────────────┐
        │ rag.llm.chat  │──── groq ─────▶│ Groq (Llama 3.1 8B) │
        │ (provider-    │                └─────────────────────┘
        │  agnostic)    │                ┌─────────────────────┐
        │               │──── ollama ───▶│ Ollama (Llama 3.2)  │
        └───────────────┘                └─────────────────────┘
                │
                ▼
        grounded answer + cited sources
```

Build-time pipeline (`python -m rag.ingest`):

```
PDFs (data/*.pdf) → page text (PyMuPDF) → chunks (size 800, overlap 100)
                                          → embeddings (bge-small-en-v1.5)
                                          → FAISS index (cosine via inner product on normalized vectors)
                                          → data/index/{faiss.index, chunks.pkl, meta.json}
```

## Tech

| Layer | Choice | Why |
|---|---|---|
| UI | Gradio | Native to Hugging Face Spaces, minimal boilerplate |
| PDF parsing | PyMuPDF (fitz) | Faster and cleaner extraction than pypdf |
| Embeddings | `BAAI/bge-small-en-v1.5` | 33M params, top-tier retrieval-per-byte, runs on CPU |
| Vector store | FAISS (CPU, `IndexFlatIP`) | In-memory, exact search, zero infra |
| LLM (cloud) | Groq, `llama-3.1-8b-instant` | Free tier, very fast, no card required |
| LLM (local) | Ollama, `llama3.2` | Fully offline, private |

No LangChain or LlamaIndex — the pipeline is small enough that hand-rolled code is clearer than a framework wrapper.

## Quickstart

### Option A — Cloud (Groq)

```bash
git clone https://github.com/ali7996/automotive-rag.git
cd automotive-rag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set GROQ_API_KEY=<your key from console.groq.com>

# Add PDFs to data/, then:
python -m rag.ingest
python app.py
```

### Option B — Local (Ollama)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2

cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama

python -m rag.ingest
python app.py
```

That's the same `app.py`. The provider switch is a single env var.

## Evaluation

`eval/run_eval.py` runs a small evaluation harness — for each question, it computes:

- **Retrieval hit-rate@k**: was the expected source document among the top-k retrieved chunks?
- **Keyword coverage**: did the LLM's answer contain the expected key terms?

```bash
python -m eval.run_eval
```

Edit `eval/questions.json` with real question/answer pairs from your own papers.

_Results table will be populated here once a real corpus is indexed._

## Project layout

```
automotive-rag/
├── app.py                # Gradio UI
├── rag/
│   ├── config.py         # Single source of truth for env config
│   ├── ingest.py         # PDFs → chunks → FAISS index
│   ├── retriever.py      # Top-k semantic search
│   ├── llm.py            # Provider-agnostic chat (Groq | Ollama)
│   └── answer.py         # Compose retrieval + LLM into a grounded answer
├── eval/
│   ├── questions.json    # Hand-written eval set
│   └── run_eval.py       # Hit-rate + keyword coverage
├── data/                 # PDFs go here (gitignored)
├── requirements.txt
└── .env.example
```

## License

MIT.
