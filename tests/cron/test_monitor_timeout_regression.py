"""Regression tests for the ctlab-doom-run-monitor API-timeout storm (t_3a2d5c8a).

Incident: a healthy monitor job on a slow/queueing reasoning provider
(qwen3.8-27b-nvfp4, thinking enabled) failed every tick with
``RuntimeError: Non-streaming API call timed out after 180s``. Each
transient timeout was booked as a hard failure, so ``failure_streak``
climbed unboundedly toward auto-disable for what is really a degraded
run, not a broken job.

Covers (with the API client mocked — the error *text* is the contract):

* timeout threshold: the qwen3.8 slug resolves to a 300s stale floor
  while the plain qwen3 family stays at 180s (longest-slug-wins).
* retry limits: consecutive transient timeouts cap ``failure_streak``
  at ``cron.transient_failure_streak_ceiling`` (default 5, 0 = uncapped).
* success path: a healthy run resets the streak.
* failure path: hard errors (auth, config, broken prompt) keep climbing
  uncapped — a genuinely broken job is never masked as a timeout storm.
* clean exit: recording a timeout outcome never raises (no unhandled
  RuntimeError escapes the cron tick).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
from cron.jobs import (
    _is_transient_api_timeout_error,
    _record_run_outcome,
    _transient_failure_streak_ceiling,
)

# The exact error text the stale-kill watchdog produced during the incident.
INCIDENT_ERROR = (
    "RuntimeError: Non-streaming API call timed out after 180s "
    "with no response (threshold: 180s)"
)


def _job(**overrides):
    job = {"id": "monitor", "failure_streak": 0, "state": "scheduled"}
    job.update(overrides)
    return job


def _run(job, **kw):
    kw.setdefault("success", False)
    kw.setdefault("error", INCIDENT_ERROR)
    kw.setdefault("delivery_error", None)
    kw.setdefault("status", None)
    kw.setdefault("now", "2026-09-07T10:00:00+00:00")
    _record_run_outcome(job, kw["success"], kw["error"],
                        kw["delivery_error"], kw["status"], kw["now"])
    return job


# ── timeout threshold: qwen3.8 floor ────────────────────────────────────────


@pytest.mark.parametrize("model", [
    "qwen3.8-27b-nvfp4",            # incident model, bare slug
    "qwen3.8-27b",
    "openrouter/qwen3.8-27b-nvfp4",  # aggregator prefix must not matter
    "local-vllm/qwen3.8-27b-nvfp4",
])
def test_qwen38_models_get_300s_floor(model):
    assert get_reasoning_stale_timeout_floor(model) == 300.0


@pytest.mark.parametrize("model,expected", [
    ("qwen/qwen3-235b-a22b-thinking", 180.0),  # plain qwen3 keeps 180s
    ("qwen/qwen3-32b", 180.0),
    ("openai/gpt-4o", None),                   # non-reasoning: no floor
    ("some-other-qwen3", None),                # slug not at start: no match
])
def test_other_models_unaffected(model, expected):
    assert get_reasoning_stale_timeout_floor(model) == expected


@pytest.mark.parametrize("bad", [None, "", 123, "   "])
def test_floor_rejects_bad_input(bad):
    assert get_reasoning_stale_timeout_floor(bad) is None


# ── transient classifier ────────────────────────────────────────────────────


@pytest.mark.parametrize("error", [
    INCIDENT_ERROR,
    "API call stale for 190s with no response",
    "stream produced no chunks for 200s",
    "TimeoutError: request timed out",
    "BrokenPipeError: [Errno 32] Broken pipe",
    "RemoteProtocolError: peer closed connection without response",
    "CUDA out of memory on the shared engine",
    "RUNTIMEERROR: NON-STREAMING API CALL TIMED OUT",  # case-insensitive
])
def test_transient_signatures_match(error):
    assert _is_transient_api_timeout_error(error) is True


@pytest.mark.parametrize("error", [
    None,
    "",
    "   ",
    "401 Unauthorized: invalid API key",          # auth stays hard
    "config drift: provider slug removed",        # drift stays hard
    "ValueError: broken prompt template",         # prompt bug stays hard
    "Job 'x' failed: exit 1",                     # generic failure stays hard
])
def test_hard_errors_do_not_match(error):
    assert _is_transient_api_timeout_error(error) is False


# ── ceiling config ──────────────────────────────────────────────────────────


def test_ceiling_defaults_to_5_without_config(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {})
    assert _transient_failure_streak_ceiling() == 5


def test_ceiling_honours_operator_override(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config",
                        lambda: {"cron": {"transient_failure_streak_ceiling": 2}})
    assert _transient_failure_streak_ceiling() == 2


def test_ceiling_falls_back_on_broken_config(monkeypatch):
    import hermes_cli.config

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(hermes_cli.config, "load_config", _boom)
    assert _transient_failure_streak_ceiling() == 5


# ── streak / retry-limit behavior ───────────────────────────────────────────


def test_success_resets_streak_and_clears_error():
    job = _run(_job(failure_streak=4), success=True, error=None)
    assert job["failure_streak"] == 0
    assert job["last_error"] is None
    assert job["last_status"] == "ok"


def test_transient_storm_caps_at_default_ceiling(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {})
    job = _job()
    for _ in range(10):
        _run(job)
    assert job["failure_streak"] == 5  # capped, not 10
    assert job["state"] == "scheduled"  # job stays enabled
    assert job["last_status"] == "error"
    assert job["last_error"] == INCIDENT_ERROR


def test_hard_failures_climb_uncapped():
    job = _job()
    for _ in range(10):
        _run(job, error="401 Unauthorized: invalid API key")
    assert job["failure_streak"] == 10


def test_ceiling_zero_means_uncapped(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config",
                        lambda: {"cron": {"transient_failure_streak_ceiling": 0}})
    job = _job()
    for _ in range(10):
        _run(job)
    assert job["failure_streak"] == 10


def test_run_claims_cleared_after_timeout():
    job = _run(_job(fire_claim={"by": "tick"}, run_claim={"by": "tick"}))
    assert job["fire_claim"] is None
    assert job["run_claim"] is None


# ── end-to-end tick with a mocked API client ────────────────────────────────


def _tick(job, api_client):
    """One monitor tick: call the (mocked) model, record the outcome."""
    try:
        api_client.complete("summarise cluster state")
    except RuntimeError as exc:
        _record_run_outcome(job, False, str(exc), None, None,
                            "2026-09-07T10:00:00+00:00")
        return "degraded"
    _record_run_outcome(job, True, None, None, None,
                        "2026-09-07T10:00:00+00:00")
    return "ok"


def test_tick_with_timeout_degrades_cleanly_without_raising():
    job = _job()
    api = Mock()
    api.complete.side_effect = RuntimeError(
        "Non-streaming API call timed out after 180s with no response "
        "(threshold: 180s)")
    assert _tick(job, api) == "degraded"  # must not propagate RuntimeError
    assert job["failure_streak"] == 1
    assert job["state"] == "scheduled"


def test_tick_with_success_resets_after_storm():
    job = _job(failure_streak=5)
    api = Mock()
    api.complete.return_value = "cluster healthy"
    assert _tick(job, api) == "ok"
    assert job["failure_streak"] == 0


def test_tick_with_hard_api_error_keeps_climbing():
    job = _job(failure_streak=5)
    api = Mock()
    api.complete.side_effect = RuntimeError("401 Unauthorized: invalid API key")
    assert _tick(job, api) == "degraded"
    assert job["failure_streak"] == 6
