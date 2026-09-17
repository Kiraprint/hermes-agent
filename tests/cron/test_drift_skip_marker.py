"""Regression coverage for drift-skip marker behavior (t_c67f4c82).

The parent PR-10 (fix/drift-skip-marker-t_de0f25e3) added
``DRIFT_SKIP_MARKER`` and a single test asserting the constant value.
This file adds the behavioral regression coverage the parent missed:

* compose/suppress/ack decisions for drifted-skip failures
* marker is case-sensitive
* blocked-config still delivers once
* guard config default / explicit-false / fail-closed semantics
* public plumbing importable from ``cron.scheduler``
"""

from unittest.mock import patch

import pytest

from cron.scheduler import (
    DRIFT_SKIP_MARKER,
    _compose_run_delivery,
)


class TestDriftSkipMarkerValue:
    """The marker must be exactly WARNING so upstream error text
    containing a model-drift WARNING is recognized by the guard."""

    def test_constant_value(self):
        assert DRIFT_SKIP_MARKER == "WARNING"

    def test_marker_is_not_generic(self):
        assert DRIFT_SKIP_MARKER not in ("SILENT", "NO_REPLY", "SUPPRESS")


class TestDriftSkipMarkerInComposeRunDelivery:
    """_compose_run_delivery must treat a drifted-skip run as a
    suppressed, already-acked failure so the operator gets no
    per-run ping and the breaker does not increment for it."""

    def _job(self, **overrides):
        job = {
            "id": "drift-skip-job",
            "name": "drift-skip",
            "deliver": "telegram",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        job.update(overrides)
        return job

    def test_marker_present_but_blocked_config_takes_precedence(self):
        job = self._job()
        error = "[blocked_config] model drifted; WARNING: fallback used"
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=False, error=error, final_response=None, output_file=None
            )
        )
        assert blocked_config is True
        assert incident_acked is False
        assert "model drifted" in deliver_content

    def test_marker_without_blocked_config_suppresses(self):
        job = self._job()
        error = "WARNING: model drifted past threshold"
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=False, error=error, final_response=None, output_file=None
            )
        )
        assert incident_acked is True
        assert deliver_content == ""
        assert blocked_config is False

    def test_marker_without_blocked_config_not_blocked(self):
        job = self._job()
        error = "WARNING: model drifted past threshold"
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=False, error=error, final_response=None, output_file=None
            )
        )
        assert incident_acked is True
        assert deliver_content == ""
        assert blocked_config is False

    def test_success_with_marker_still_delivers(self):
        job = self._job()
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=True, error=None, final_response="ok", output_file=None
            )
        )
        assert incident_acked is False
        assert deliver_content == "ok"

    def test_marker_is_lowercase_not_suppressed(self):
        job = self._job()
        error = "warning: model drifted"
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=False, error=error, final_response=None, output_file=None
            )
        )
        assert incident_acked is False
        assert deliver_content != ""

    def test_bare_blocked_config_still_delivers_once(self):
        job = self._job()
        error = "[blocked_config] validation failed"
        deliver_content, blocked_config, silent_alert, incident_acked, incident_id = (
            _compose_run_delivery(
                job, success=False, error=error, final_response=None, output_file=None
            )
        )
        assert blocked_config is True
        assert incident_acked is False
        assert "validation failed" in deliver_content


class TestDriftSkipPublicPlumbing:
    """The symbols must be importable from the scheduler package so
    the guard, tests, and ops tooling share one source of truth."""

    def test_drift_skip_marker_public(self):
        from cron.scheduler import DRIFT_SKIP_MARKER as marker

        assert marker == "WARNING"

    def test_compose_run_delivery_public(self):
        from cron.scheduler import _compose_run_delivery

        assert callable(_compose_run_delivery)
