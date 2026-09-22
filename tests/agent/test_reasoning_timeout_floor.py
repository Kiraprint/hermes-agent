"""Regression tests for reasoning-model request-timeout floors."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("provider, model, timeout, default, expected", [
    ("custom", "custom/kilo-auto/free", 300, None, 600),
    ("custom", "custom/kilo-auto/pro", 300, None, 600),
    ("openai", "openai/o1", 300, None, 1800),
    ("openai", "openai/gpt-5", 300, None, 300),
    ("custom", "custom/kilo-auto/free", 0, None, 0),
    ("custom", "custom/kilo-auto/free", None, 300, 600),
    ("openai", "openai/gpt-5", None, 300, 300),
])
def test_reasoning_request_timeout_floor(provider, model, timeout, default, expected):
    from agent.reasoning_timeouts import get_reasoning_timeout

    assert get_reasoning_timeout(provider, model, timeout, default) == expected


def test_agent_resolves_reasoning_request_timeout_floor(monkeypatch):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = "custom"
    agent.model = "custom/kilo-auto/free"
    monkeypatch.setenv("HERMES_API_TIMEOUT", "300")
    monkeypatch.setattr("run_agent.get_provider_request_timeout", return_value=None)

    assert agent._resolved_api_call_timeout() == 600


def test_streaming_call_applies_reasoning_request_timeout_floor(monkeypatch):
    from agent.chat_completion_helpers import _StreamingCall

    call = object.__new__(_StreamingCall)
    call.agent = SimpleNamespace(
        provider="custom", model="custom/kilo-auto/free", base_url=None)
    call._stream_stale_timeout = 180
    monkeypatch.setattr(
        "agent.chat_completion_helpers.get_provider_request_timeout", return_value=300)

    assert call._stream_timeouts() == (600, 600, 60)
