"""I-13 — provider-chokepoint retry/backoff on transient LLM errors.

A single transient 429/529/5xx/timeout otherwise crashes a whole shard run mid-pipeline. Retry lives
in the base ``Provider.complete`` so every caller is covered; non-transient errors are not retried.
"""

import time

import pytest

from scpilot.llm.provider import (LLMResponse, Provider, ProviderConfig, ProviderError,
                                   ProviderUnavailable, _is_retryable)


class _FlakyProvider(Provider):
    """Fails its first ``fail_times`` calls with ``exc``, then returns ok."""

    def __init__(self, fail_times, exc):
        super().__init__(ProviderConfig(backend="anthropic", model="m"))
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0

    def _complete(self, messages, *, tools=None, system=None, tool_choice=None, max_tokens=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return LLMResponse(text="ok")


def test_is_retryable_classification():
    assert _is_retryable(ProviderError("Error 529: overloaded"))
    assert _is_retryable(ProviderError("rate limit exceeded (429)"))
    assert _is_retryable(ProviderError("connection reset"))
    assert not _is_retryable(ProviderError("401 authentication_error"))
    assert not _is_retryable(ProviderError("400 invalid_request_error"))
    assert not _is_retryable(ProviderUnavailable("no ANTHROPIC_API_KEY"))   # permanent


def test_retries_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    p = _FlakyProvider(3, ProviderError("overloaded 529"))
    resp = p.complete([{"role": "user", "content": "x"}])
    assert resp.text == "ok"
    assert p.calls == 4                       # 3 failures + 1 success
    assert len(slept) == 3 and slept == [2.0, 4.0, 8.0]   # exponential backoff schedule


def test_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    p = _FlakyProvider(99, ProviderError("503 server error"))
    p.max_retries = 2
    with pytest.raises(ProviderError):
        p.complete([{"role": "user", "content": "x"}])
    assert p.calls == 3                        # initial + 2 retries, then re-raise


def test_non_retryable_raises_immediately(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    p = _FlakyProvider(99, ProviderError("400 invalid_request bad schema"))
    with pytest.raises(ProviderError):
        p.complete([{"role": "user", "content": "x"}])
    assert p.calls == 1                        # no retry on a permanent error


def test_backoff_is_capped(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    p = _FlakyProvider(6, ProviderError("timeout"))
    p.max_retries = 6
    p.complete([{"role": "user", "content": "x"}])
    assert max(slept) == 30.0                  # backoff_cap honored (2**5=32 → 30)
