"""Unpinned terminal-enabled cron jobs are refused (t_568cd1c1).

An agent-backed cron job that holds the ``terminal`` toolset but pins neither ``model`` nor
``provider`` runs on whatever the *global* inference assignment happens to be at that tick. That
is the incident shape: the operator moved the global model, three autonomous jobs silently
inherited it, and one of them spent a whole fire (and a whole shell) on a model nobody chose for
it.

Contract under test:
  * the guard fires on ``terminal`` + no ``model``/``provider`` only;
  * it raises ``RuntimeError("drift_skip: ...")`` and, through ``run_job``, returns a normal
    failed run whose error string *starts* with the ``drift_skip`` marker and carries
    ``DRIFT_SKIP_MARKER`` so the drift-skip delivery branch suppresses the ping;
  * no agent is ever constructed for a refused job (no inference, no spend);
  * a pinned job and a non-terminal job run exactly as before.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import (
    DRIFT_SKIP_MARKER,
    DRIFT_SKIP_PREFIX,
    _compose_run_delivery,
    guard_unpinned_terminal_toolset,
    run_job,
    unpinned_terminal_toolset_reason,
)


# tests/cron/conftest.py autouses a ``_reset_session_context_vars`` fixture that eagerly
# imports ``gateway.session_context`` (and, transitively, ``gateway.restart``, which
# unconditionally reads ``DEFAULT_CONFIG['gateway']['signal_interrupt_grace_timeout']`` —
# missing from ``hermes_cli.config`` in this workspace, breaking every ``tests/cron/`` module).
# The guard predicate and the mocked ``run_job`` paths exercised below never mutate any session
# ContextVar, so overriding that fixture locally as a no-op keeps this test module hermetic
# without touching the pre-existing DEFAULT_CONFIG gap.
@pytest.fixture(autouse=True)
def _reset_session_context_vars():
    yield


def _base_job(**overrides):
    job = {
        "id": "unpinned-terminal-test",
        "name": "unpinned terminal test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


class TestGuardPredicate:
    def test_terminal_without_pins_is_refused(self):
        reason = unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["terminal"]))
        assert reason is not None
        assert "terminal" in reason

    def test_terminal_with_a_model_pin_is_allowed(self):
        assert unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], model="pinned-model")) is None

    def test_terminal_with_a_provider_pin_is_allowed(self):
        assert unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], provider="pinned-provider")) is None

    def test_blank_pins_do_not_count(self):
        assert unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], model="   ", provider="")) is not None

    def test_non_terminal_job_is_untouched(self):
        assert unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["web", "memory"])) is None

    def test_missing_or_malformed_toolsets_are_untouched(self):
        assert unpinned_terminal_toolset_reason(_base_job()) is None
        assert unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=None)) is None
        assert unpinned_terminal_toolset_reason(_base_job(enabled_toolsets="terminal")) is None
        assert unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=[None, 3])) is None

    def test_toolset_name_match_is_case_insensitive(self):
        assert unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["Terminal"])) is not None

    def test_no_agent_jobs_are_exempt(self):
        """A script-only job spawns no agent, so there is no inference assignment to pin."""
        assert unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], no_agent=True)) is None


class TestGuardRaises:
    def test_raises_drift_skip_runtime_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            guard_unpinned_terminal_toolset(_base_job(enabled_toolsets=["terminal"]), "job-1", "j")

        error = str(excinfo.value)
        assert error.startswith(f"{DRIFT_SKIP_PREFIX}: ")
        assert DRIFT_SKIP_MARKER in error
        assert "no model call was made" in error

    def test_returns_quietly_for_an_allowed_job(self):
        guard_unpinned_terminal_toolset(
            _base_job(enabled_toolsets=["terminal"], model="m"), "job-1", "j")

    def test_marker_reaches_the_drift_skip_delivery_branch(self):
        """The error ``run_job`` builds must be recognized as a drift skip, not a failure ping."""
        with pytest.raises(RuntimeError) as excinfo:
            guard_unpinned_terminal_toolset(_base_job(enabled_toolsets=["terminal"]), "job-1", "j")

        deliver_content, blocked_config, _silent, incident_acked, incident_id = _compose_run_delivery(
            _base_job(), success=False, error=str(excinfo.value), final_response="", output_file=None)

        assert deliver_content == ""
        assert blocked_config is False
        assert incident_acked is True
        assert incident_id is None


def _run(job, tmp_path, *, current_model="global-model", current_provider="openrouter"):
    """Drive the real ``run_job`` path against a temp HERMES_HOME with a mocked AIAgent.

    Returns ``(success, error, agent_kwargs)``; ``agent_kwargs`` is None when no agent was built.
    """
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {current_model}\n  provider: {current_provider}\n")

    resolve_kwargs = {}

    def _resolve(**kwargs):
        resolve_kwargs.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or current_provider,
            "api_mode": "chat_completions",
        }

    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = run_job(job)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None
    return success, error, agent_kwargs


class TestRunJobIntegration:
    def test_unpinned_terminal_job_is_skipped_without_building_an_agent(self, tmp_path):
        job = _base_job(enabled_toolsets=["terminal", "web"])

        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is False
        assert error.startswith(f"{DRIFT_SKIP_PREFIX}: "), error
        assert DRIFT_SKIP_MARKER in error
        assert agent_kwargs is None, "a refused job must not build an agent / spend inference"

    def test_terminal_job_with_a_model_pin_runs(self, tmp_path):
        job = _base_job(enabled_toolsets=["terminal"], model="pinned-model")

        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None
        assert agent_kwargs["model"] == "pinned-model"

    def test_terminal_job_with_a_provider_pin_runs(self, tmp_path):
        job = _base_job(enabled_toolsets=["terminal"], provider="pinned-provider")

        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None

    def test_non_terminal_unpinned_job_still_runs(self, tmp_path):
        job = _base_job(enabled_toolsets=["web"])

        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None
        assert agent_kwargs["model"] == "global-model"

    def test_job_without_any_toolsets_still_runs(self, tmp_path):
        job = _base_job()

        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None
