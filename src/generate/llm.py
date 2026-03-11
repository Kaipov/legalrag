"""
LLM interface with streaming for answer generation.

Uses OpenAI-compatible APIs with streaming for TTFT optimization.
Includes retry/backoff for transient rate limits and provider hiccups.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Generator

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.config import get_llm_api_key, get_llm_api_base, GENERATION_MODEL, GENERATION_TEMPERATURE

logger = logging.getLogger(__name__)

# Lazy-initialized client
_client: OpenAI | None = None
_MAX_RETRIES = 4
_INITIAL_RETRY_DELAY_S = 2.0
_MAX_RETRY_DELAY_S = 20.0
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _get_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        api_key = get_llm_api_key()
        api_base = get_llm_api_base()
        if not api_key:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
        _client = OpenAI(api_key=api_key, base_url=api_base)
        logger.info("OpenAI client initialized (base=%s, model=%s)", api_base, GENERATION_MODEL)
    return _client


def _get_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    return None


def _get_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError)):
        return True

    if isinstance(exc, APIError):
        status_code = _get_status_code(exc)
        return status_code in _RETRYABLE_STATUS_CODES

    status_code = _get_status_code(exc)
    return status_code in _RETRYABLE_STATUS_CODES


def _compute_retry_delay(exc: Exception, attempt: int) -> float:
    retry_after = _get_retry_after_seconds(exc)
    if retry_after is not None:
        return min(_MAX_RETRY_DELAY_S, retry_after)

    base_delay = min(_MAX_RETRY_DELAY_S, _INITIAL_RETRY_DELAY_S * (2 ** attempt))
    jitter = random.uniform(0.0, max(0.1, base_delay * 0.25))
    return min(_MAX_RETRY_DELAY_S, base_delay + jitter)


def _stream_once(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
) -> Generator[str, None, None]:
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

    attempt = 0
    while True:
        yielded_any = False
        try:
            for token in _stream_once(client, messages, model, temperature):
                yielded_any = True
                yield token
            return
        except Exception as exc:
            if yielded_any or attempt >= _MAX_RETRIES or not _is_retryable(exc):
                raise

            delay = _compute_retry_delay(exc, attempt)
            status_code = _get_status_code(exc)
            logger.warning(
                "Transient LLM error (%s) on attempt %s/%s; retrying in %.1fs",
                status_code or type(exc).__name__,
                attempt + 1,
                _MAX_RETRIES + 1,
                delay,
            )
            time.sleep(delay)
            attempt += 1


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