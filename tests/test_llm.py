from __future__ import annotations

from types import SimpleNamespace

from src.generate import llm as llm_mod


class _FakeError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"status={status_code}")
        headers = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


class _FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs_history.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
    )


def test_stream_generate_retries_on_rate_limit(monkeypatch) -> None:
    completions = _FakeCompletions([
        _FakeError(429, retry_after="0"),
        [_chunk("hello"), _chunk(" world")],
    ])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(llm_mod, "_get_client", lambda: fake_client)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_args, **_kwargs: None)

    output = "".join(llm_mod.stream_generate([{"role": "user", "content": "hi"}]))

    assert output == "hello world"
    assert completions.calls == 2


def test_compute_retry_delay_prefers_retry_after_header() -> None:
    delay = llm_mod._compute_retry_delay(_FakeError(429, retry_after="7"), attempt=0)

    assert delay == 7


def test_build_create_kwargs_uses_gpt5_compatible_parameters() -> None:
    kwargs = llm_mod._build_create_kwargs(
        [{"role": "user", "content": "hi"}],
        "gpt-5-mini",
        0.1,
    )

    assert kwargs["max_completion_tokens"] == 500
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_build_create_kwargs_uses_legacy_chat_parameters() -> None:
    kwargs = llm_mod._build_create_kwargs(
        [{"role": "user", "content": "hi"}],
        "gpt-4.1-mini",
        0.1,
    )

    assert kwargs["max_tokens"] == 500
    assert "max_completion_tokens" not in kwargs
    assert kwargs["temperature"] == 0.1
