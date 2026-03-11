"""
LLM interface with streaming for answer generation.

Uses OpenAI API (gpt-4o) with streaming for TTFT optimization.
"""
from __future__ import annotations

import logging
import time
from typing import Generator

from openai import OpenAI

from src.config import get_llm_api_key, get_llm_api_base, GENERATION_MODEL, GENERATION_TEMPERATURE

logger = logging.getLogger(__name__)

# Lazy-initialized client
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        api_key = get_llm_api_key()
        api_base = get_llm_api_base()
        if not api_key:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
        _client = OpenAI(api_key=api_key, base_url=api_base)
        logger.info(f"OpenAI client initialized (base={api_base}, model={GENERATION_MODEL})")
    return _client


def stream_generate(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
) -> Generator[str, None, None]:
    """
    Stream tokens from the LLM.

    Yields individual text chunks as they arrive.
    Use with TelemetryTimer.mark_token() for TTFT tracking.
    """
    client = _get_client()
    model = model or GENERATION_MODEL
    temperature = temperature if temperature is not None else GENERATION_TEMPERATURE

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        max_tokens=500,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Non-streaming generation. Returns the full response text.
    Use stream_generate() for TTFT-optimized pipeline.
    """
    chunks = list(stream_generate(messages, model, temperature))
    return "".join(chunks)
