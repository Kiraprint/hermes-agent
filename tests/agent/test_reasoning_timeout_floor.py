"""Regression tests for the reasoning-model per-request timeout floor (t_568cd1c1).

Before the floor, an operator/global ``request_timeout_seconds`` (300s in the incident) was
honored verbatim for reasoning models, so a call that legitimately thinks for minutes was cut
off mid-think. The floor is applied by :func:`agent.reasoning_timeouts.get_reasoning_timeout`
and reached through both request paths:

* ``AIAgent._resolved_api_call_timeout()`` — the value both the streaming and the non-streaming
  wire hand to the transport;
* ``_StreamingCall._stream_timeouts()`` — the raw socket timeouts of the streaming wire.

Preserved contracts: an explicit ``0`` disables the timeout, ``None`` falls back to the
provider/model default chain, and a plain (non-reasoning) model keeps its configured value.
"""

from types import SimpleNamespace

import pytest

from agent.reasoning_timeouts import (
    get_reasoning_stale_timeout_floor,
    get_reasoning_timeout,
)


class TestGetReasoningTimeout:
    """Unit contract of the floor resolver."""

    @pytest.mark.parametrize("provider, model, timeout, default, expected", [
        # kilo-auto aggregator: 600s floor, even over an explicit sub-floor 300s.
        ("custom", "custom/kilo-auto/free", 300, None, 600),
        ("custom", "custom/kilo-auto/pro", 300, None, 600),
        # Any other allowlisted reasoning model: 1800s floor.
        ("openai", "openai/o1", 300, None, 1800),
        ("openai", "openai/o3-mini", 300, None, 1800),
        # Plain model: the operator's value is honored verbatim (no floor).
        ("openai", "openai/gpt-5", 300, None, 300),
        # A value at or above the floor is never lowered.
        ("custom", "custom/kilo-auto/free", 900, None, 900),
        ("openai", "openai/o1", 3600, None, 3600),
        # Explicit 0 = timeout disabled; the floor never re-enables it.
        ("custom", "custom/kilo-auto/free", 0, None, 0),
        ("openai", "openai/o1", 0, None, 0),
        # None = nothing configured: the provider/model default, still raised to the floor.
        ("custom", "custom/kilo-auto/free", None, 300, 600),
        ("openai", "openai/o1", None, 300, 1800),
        ("openai", "openai/gpt-5", None, 300, 300),
        # Malformed timeout values fall through to the default instead of exploding.
        ("custom", "custom/kilo-auto/free", "not-a-number", 300, 600),
    ])
    def test_floor_contract(self, provider, model, timeout, default, expected):
        assert get_reasoning_timeout(provider, model, timeout, default) == expected

    def test_none_without_default_is_1800(self):
        """``None`` with no resolved default chain still lands on the 1800s built-in."""
        assert get_reasoning_timeout("openai", "openai/gpt-5", None, None) == 1800
        assert get_reasoning_timeout("custom", "custom/kilo-auto/free", None, None) == 1800

    def test_kilo_auto_router_detection_is_component_based(self):
        """Any ``/``-separated ``kilo-auto*`` component matches, so future SKUs keep the floor."""
        assert get_reasoning_timeout("custom", "kilo-auto/free", 300, None) == 600
        assert get_reasoning_timeout("custom", "kilo-auto/future-sku", 300, None) == 600
        # Not a kilo-auto component -> no 600s floor (plain model keeps its value).
        assert get_reasoning_timeout("custom", "custom/kilo-pro", 300, None) == 300

    def test_floor_is_additive_to_the_stale_timeout_allowlist(self):
        """The request floor must not change the (separate) stale-detector allowlist."""
        assert get_reasoning_stale_timeout_floor("custom/kilo-auto/free") is None
        assert get_reasoning_stale_timeout_floor("openai/o1") == 600


class TestAgentRequestTimeoutFloor:
    """``_resolved_api_call_timeout`` is what both wires pass to the transport."""

    def _agent(self, provider, model):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        return agent

    def test_env_sub_floor_timeout_is_raised_for_kilo_auto(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_TIMEOUT", "300")
        monkeypatch.setattr("run_agent.get_provider_request_timeout", lambda *a, **k: None)

        assert self._agent("custom", "custom/kilo-auto/free")._resolved_api_call_timeout() == 600

    def test_env_sub_floor_timeout_is_raised_for_other_reasoning_models(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_TIMEOUT", "300")
        monkeypatch.setattr("run_agent.get_provider_request_timeout", lambda *a, **k: None)

        assert self._agent("openai", "openai/o1")._resolved_api_call_timeout() == 1800

    def test_plain_model_keeps_its_configured_timeout(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_TIMEOUT", "300")
        monkeypatch.setattr("run_agent.get_provider_request_timeout", lambda *a, **k: 42.0)

        assert self._agent("openrouter", "openai/gpt-4o-mini")._resolved_api_call_timeout() == 42.0

    def test_reasoning_model_with_nothing_configured_defaults_to_1800(self, monkeypatch):
        monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
        monkeypatch.setattr("run_agent.get_provider_request_timeout", lambda *a, **k: None)

        assert self._agent("openai", "openai/o1")._resolved_api_call_timeout() == 1800

    def test_explicit_zero_stays_disabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
        monkeypatch.setattr("run_agent.get_provider_request_timeout", lambda *a, **k: 0.0)

        assert self._agent("custom", "custom/kilo-auto/free")._resolved_api_call_timeout() == 0.0


class TestStreamingSocketTimeoutFloor:
    """``_stream_timeouts`` is the streaming wire's raw socket budget."""

    def _call(self, provider, model):
        from agent.chat_completion_helpers import _StreamingCall

        call = object.__new__(_StreamingCall)
        call.agent = SimpleNamespace(provider=provider, model=model, base_url=None)
        call._stream_stale_timeout = 180
        return call

    def test_sub_floor_configured_timeout_is_raised_for_kilo_auto(self, monkeypatch):
        monkeypatch.setattr(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            lambda *a, **k: 300)

        assert self._call("custom", "custom/kilo-auto/free")._stream_timeouts() == (600, 600, 60)

    def test_sub_floor_configured_timeout_is_raised_for_other_reasoning_models(self, monkeypatch):
        monkeypatch.setattr(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            lambda *a, **k: 300)

        assert self._call("openai", "openai/o1")._stream_timeouts() == (1800, 1800, 60)

    def test_plain_model_keeps_configured_socket_timeouts(self, monkeypatch):
        monkeypatch.setattr(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            lambda *a, **k: 300)

        assert self._call("openrouter", "openai/gpt-4o-mini")._stream_timeouts() == (300, 300, 60)

    def test_unconfigured_reasoning_model_keeps_1800_write_budget(self, monkeypatch):
        monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
        monkeypatch.setattr(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            lambda *a, **k: None)

        call = self._call("openai", "openai/o1")
        # As resolved in a real run: the reasoning stale detector's 600s floor for o1.
        call._stream_stale_timeout = 600
        write, read, connect = call._stream_timeouts()
        assert write == 1800
        # The stale detector's reasoning floor raises the read budget above the 120s default.
        assert read == 600
        assert connect == 30.0
