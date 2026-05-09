"""Gradio chat UI for the automotive RAG."""

from __future__ import annotations

import gradio as gr

from rag.answer import answer
from rag.llm import active_backend


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        snippet = s["text"][:300].replace("\n", " ")
        lines.append(
            f"**[{i}] {s['doc_id']} · page {s['page']}** — score {s['score']:.3f}\n\n> {snippet}…"
        )
    return "\n\n".join(lines)


def respond(question: str, history: list[dict]):
    if not question.strip():
        return history, ""
    history = history + [{"role": "user", "content": question}]
    try:
        result = answer(question)
        reply = result["answer"]
        sources = _format_sources(result["sources"])
        full = f"{reply}\n\n---\n**Sources**\n\n{sources}"
    except FileNotFoundError as e:
        full = f"⚠️ {e}"
    except Exception as e:
        full = f"⚠️ Error: {type(e).__name__}: {e}"
    history = history + [{"role": "assistant", "content": full}]
    return history, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Automotive RAG") as demo:
        gr.Markdown(
            f"# 🚗 Automotive RAG\n"
            f"Ask questions about automotive engineering papers. "
            f"**Backend:** `{active_backend()}`"
        )
        chat_box = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Question", placeholder="e.g. How does end-to-end learning differ from modular pipelines?")
        clear = gr.Button("Clear")
        msg.submit(respond, [msg, chat_box], [chat_box, msg])
        clear.click(lambda: ([], ""), outputs=[chat_box, msg])
    return demo


if __name__ == "__main__":
    build_app().launch()
