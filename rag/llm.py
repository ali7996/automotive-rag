"""Provider-agnostic LLM. Set LLM_PROVIDER=groq or LLM_PROVIDER=ollama."""

from __future__ import annotations

import requests

from rag.config import settings


def _chat_groq(messages: list[dict], temperature: float) -> str:
    from groq import Groq

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    client = Groq(api_key=settings.groq_api_key)
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _chat_ollama(messages: list[dict], temperature: float) -> str:
    resp = requests.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    if settings.llm_provider == "groq":
        return _chat_groq(messages, temperature)
    if settings.llm_provider == "ollama":
        return _chat_ollama(messages, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def active_backend() -> str:
    if settings.llm_provider == "groq":
        return f"Groq · {settings.groq_model}"
    if settings.llm_provider == "ollama":
        return f"Ollama · {settings.ollama_model}"
    return settings.llm_provider
