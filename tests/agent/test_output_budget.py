"""Outbound output cap must fit the window the prompt already occupies (agent/output_budget).

The incident this guards: local-vllm (qwen3.8-27b-nvfp4, context 190000) was asked for
124465 input + 65536 output = 190001 tokens and answered HTTP 400 — the request was doomed
before it left the process, and the refusal took the bg-review and compression lanes with
it. ``max_tokens`` is a reservation on the same window as the prompt, so the cap has to be
derived per request, not trusted as a constant.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.output_budget import (
    MIN_OUTPUT_TOKENS,
    OUTPUT_WINDOW_HEADROOM_TOKENS,
    OutputWindowExhausted,
    apply_output_window_guard,
    available_output_tokens,
    clamp_output_tokens,
    output_cap_field,
)

WINDOW = 190000
INCIDENT_PROMPT_TOKENS = 124465
INCIDENT_REQUESTED = 65536


def _stub_agent(context_length=WINDOW):
    """Minimal carrier of the one attribute the guard reads."""
    return SimpleNamespace(context_compressor=SimpleNamespace(context_length=context_length))


# ── pure arithmetic ─────────────────────────────────────────────────────

def test_available_output_tokens_keeps_headroom():
    assert available_output_tokens(WINDOW, INCIDENT_PROMPT_TOKENS) == WINDOW - OUTPUT_WINDOW_HEADROOM_TOKENS - INCIDENT_PROMPT_TOKENS
    # A prompt past the window never yields a negative budget.
    assert available_output_tokens(WINDOW, WINDOW + 10) == 0


@pytest.mark.parametrize(
    "requested,available,expected",
    [(65536, 64511, 64511), (4096, 64511, 4096), (65536, MIN_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS)],
)
def test_clamp_output_tokens_takes_the_smaller(requested, available, expected):
    assert clamp_output_tokens(requested, available) == expected


def test_clamp_refuses_when_no_usable_answer_fits():
    assert clamp_output_tokens(1, MIN_OUTPUT_TOKENS - 1) is None


def test_output_cap_field_prefers_the_transport_specific_key():
    assert output_cap_field({"max_tokens": 100, "max_completion_tokens": 200}) == ("max_completion_tokens", 200)
    assert output_cap_field({"model": "m"}) == (None, None)


# ── the guard ───────────────────────────────────────────────────────────

def test_incident_request_is_clamped_inside_the_window():
    """124465 input + 65536 output = 190001 (HTTP 400) must become a request that fits."""
    api_kwargs = {"model": "qwen3.8-27b-nvfp4", "max_tokens": INCIDENT_REQUESTED}

    cap = apply_output_window_guard(_stub_agent(), api_kwargs, prompt_tokens=INCIDENT_PROMPT_TOKENS)

    assert api_kwargs["max_tokens"] == cap == WINDOW - OUTPUT_WINDOW_HEADROOM_TOKENS - INCIDENT_PROMPT_TOKENS
    assert INCIDENT_PROMPT_TOKENS + cap <= WINDOW - OUTPUT_WINDOW_HEADROOM_TOKENS


def test_a_cap_that_already_fits_is_left_untouched():
    """The guard is no-regret: it only rewrites caps that would have been rejected anyway."""
    api_kwargs = {"max_tokens": 4096}

    assert apply_output_window_guard(_stub_agent(), api_kwargs, prompt_tokens=1000) == 4096
    assert api_kwargs["max_tokens"] == 4096


def test_request_without_a_cap_is_untouched():
    """No cap → the provider fits the answer itself; nothing to clamp, nothing to reject."""
    api_kwargs = {"model": "m"}

    assert apply_output_window_guard(_stub_agent(), api_kwargs, prompt_tokens=999_999) is None
    assert api_kwargs == {"model": "m"}


def test_unknown_window_is_untouched():
    api_kwargs = {"max_tokens": INCIDENT_REQUESTED}

    assert apply_output_window_guard(SimpleNamespace(), api_kwargs, prompt_tokens=999_999) == INCIDENT_REQUESTED
    assert api_kwargs["max_tokens"] == INCIDENT_REQUESTED


def test_prompt_that_ate_the_window_is_never_sent():
    """The whole point: refuse locally instead of collecting a provider 400."""
    with pytest.raises(OutputWindowExhausted) as exc:
        apply_output_window_guard(
            _stub_agent(), {"max_tokens": INCIDENT_REQUESTED}, prompt_tokens=WINDOW - MIN_OUTPUT_TOKENS
        )

    assert f"{MIN_OUTPUT_TOKENS}" in str(exc.value)
    assert f"{WINDOW:,}" in str(exc.value)


# ── recovery ────────────────────────────────────────────────────────────

def test_refusal_classifies_as_context_overflow():
    """A refused request must compress, not retry the identical doomed call."""
    from agent.error_classifier import FailoverReason, classify_api_error

    verdict = classify_api_error(
        OutputWindowExhausted(f"Prompt of ~{WINDOW - 10:,} tokens leaves less than {MIN_OUTPUT_TOKENS} tokens"),
        provider="local-vllm",
        model="qwen3.8-27b-nvfp4",
        approx_tokens=WINDOW - 10,
        context_length=WINDOW,
        num_messages=12,
    )

    assert verdict.reason is FailoverReason.context_overflow
    assert verdict.should_compress is True


# ── wiring ──────────────────────────────────────────────────────────────

@pytest.fixture()
def local_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://localhost:8000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.context_compressor.context_length = WINDOW
        agent.max_tokens = INCIDENT_REQUESTED
        return agent


def test_build_api_kwargs_derives_the_cap_from_the_prompt(local_agent):
    """The guard runs on the real request build, on the prompt that build carries."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    messages = [{"role": "user", "content": "x" * (INCIDENT_PROMPT_TOKENS * 4)}]
    api_kwargs = local_agent._build_api_kwargs(messages)

    field, cap = output_cap_field(api_kwargs)
    assert field is not None, "the wire request must carry an outbound cap"
    # The prompt the guard measured: the same payload before the cap was rewritten.
    est_prompt = estimate_request_context_tokens({**api_kwargs, field: INCIDENT_REQUESTED})

    assert cap is not None and cap < INCIDENT_REQUESTED
    assert cap == WINDOW - OUTPUT_WINDOW_HEADROOM_TOKENS - est_prompt
    assert est_prompt + cap <= WINDOW - OUTPUT_WINDOW_HEADROOM_TOKENS
