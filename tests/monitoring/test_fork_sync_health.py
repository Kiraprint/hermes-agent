"""Tests for the fork-sync divergence feed (``agent/monitoring/fork_sync_health``).

The feed is observational: it must classify a healthy / degraded / unhealthy
fork-sync runner state from the runner's own artefacts (its log + the board
escalation task it opens), and every broken read must degrade the *reported
status* without raising into the export loop, the readiness probe or the
gateway watcher.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone

import pytest

from agent.monitoring import fork_sync_health as fsh

NOW = 1_800_000_000.0  # fixed clock: ages are asserted, never wall-clock raced


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_log(tmp_path, *messages, epoch: float = NOW, name: str = "fork-sync.log"):
    """Write a runner log whose lines carry the real ``[ISO-Z]`` prefix."""
    path = tmp_path / name
    path.write_text(
        "".join(f"[{_stamp(epoch)}] {msg}\n" for msg in messages), encoding="utf-8"
    )
    return path


def _board_db(tmp_path, rows, name: str = "kanban.db"):
    """Minimal board DB carrying just the columns the escalation probe reads."""
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT, idempotency_key TEXT)"
    )
    conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture(autouse=True)
def _clean_cache():
    """The snapshot is cached module-globally (shared with the gateway watcher)."""
    fsh.reset_fork_sync_health_cache()
    yield
    fsh.reset_fork_sync_health_cache()


@pytest.fixture()
def no_board(monkeypatch):
    """Keep the escalation probe off the ambient board unless a test wants it."""
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", lambda board=None, **kw: 0)


def _board_reader(db_path):
    """Point the escalation probe at a fixture DB (never at the ambient board).

    The real function is captured first: ``fsh.count_open_escalation_tasks`` is
    the monkeypatched attribute, so calling it from the replacement recurses.
    """
    real = fsh.count_open_escalation_tasks

    def _read(board=None, **kwargs):
        return real(board, db_path=db_path)

    return _read


def _read(tmp_path, log_path, **kwargs) -> fsh.ForkSyncState:
    settings = fsh.ForkSyncSettings(
        log_path=log_path,
        stale_after_seconds=kwargs.pop("stale_after_seconds", fsh.DEFAULT_STALE_AFTER_SECONDS),
    )
    return fsh.read_fork_sync_state(settings, now=kwargs.pop("now", NOW))


# --------------------------------------------------------------------------- statuses


def test_fresh_clean_sync_is_healthy(tmp_path, no_board):
    log = _write_log(tmp_path, "RESULT clean sync fork/main -> abc1234 (up to date; round 0) (rc=0)")

    state = _read(tmp_path, log)

    assert state.status == fsh.HEALTHY
    assert state.reason == "clean_sync"
    assert state.last_result == "clean"
    assert state.last_exit_code == 0
    assert state.last_success_age_seconds == pytest.approx(0.0)


def test_stale_clean_sync_is_degraded(tmp_path, no_board):
    log = _write_log(
        tmp_path,
        "RESULT clean sync fork/main -> abc1234 (up to date; round 0) (rc=0)",
        epoch=NOW - 40 * 3600,  # 40h > 36h default grace
    )

    state = _read(tmp_path, log)

    assert state.status == fsh.DEGRADED
    assert state.reason == "stale_sync"
    assert state.last_success_age_seconds == pytest.approx(40 * 3600)


def test_later_failure_after_an_earlier_success_is_unhealthy(tmp_path, no_board):
    """A stale success must not mask the newest run's failure."""
    log = _write_log(
        tmp_path,
        "RESULT clean sync fork/main -> abc1234 (up to date; round 0) (rc=0)",
        "RESULT push failed fork/main -> def5678 (rc=3)",
    )

    state = _read(tmp_path, log)

    assert state.status == fsh.UNHEALTHY
    assert state.reason == "push_failed"
    assert state.last_exit_code == 3
    assert state.last_success_age_seconds == pytest.approx(0.0)


def test_failed_run_is_unhealthy(tmp_path, no_board):
    log = _write_log(tmp_path, "RESULT error fetch upstream failed (rc=1)")

    state = _read(tmp_path, log)

    assert state.status == fsh.UNHEALTHY
    assert state.reason == "run_failed"
    assert state.last_exit_code == 1


def test_conflict_with_open_escalation_is_degraded(tmp_path, monkeypatch):
    log = _write_log(
        tmp_path,
        "RESULT created task t_ab12cd (conflict escalated: 2 files; round 3) (rc=2)",
    )
    db_path = _board_db(tmp_path, [
        ("t_ab12cd", "Fork-sync конфликт: hermes_cli/kanban_db.py", "blocked", "fork-sync-conflict-abc"),
    ])
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", _board_reader(db_path))

    state = _read(tmp_path, log)

    assert state.status == fsh.DEGRADED
    assert state.reason == "conflict_escalated"
    assert state.escalation_open == 1
    assert state.board_read_failed is False
    assert state.last_exit_code == 2


def test_conflict_without_open_escalation_is_unhealthy(tmp_path, monkeypatch):
    """A conflict whose escalation task is already closed is nobody's work: unhealthy."""
    log = _write_log(tmp_path, "RESULT created task t_ab12cd (conflict escalated; rc=2)")
    db_path = _board_db(tmp_path, [
        ("t_ab12cd", "Fork-sync конфликт: old", "done", "fork-sync-conflict-abc"),
        ("t_ffff99", "Unrelated task", "ready", "some-other-key"),
    ])
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", _board_reader(db_path))

    state = _read(tmp_path, log)

    assert state.status == fsh.UNHEALTHY
    assert state.reason == "conflict_unattended"
    assert state.escalation_open == 0


def test_legacy_runner_lines_are_recognised(tmp_path, no_board):
    """Pre-hardening script wrote free-form lines; a host mid-migration still reports."""
    log = _write_log(tmp_path, "ERROR: rebase ok but push failed (branch fork/main)")

    state = _read(tmp_path, log)

    assert state.status == fsh.UNHEALTHY
    assert state.reason == "push_failed"
    assert state.last_exit_code == 3  # inferred from the outcome, no explicit (rc=N)


def test_legacy_conflict_lines_are_recognised(tmp_path, no_board):
    log = _write_log(tmp_path, "CONFLICT: hermes_cli/kanban_db.py — opening factory task")

    state = _read(tmp_path, log)

    assert state.last_result == "conflict"
    assert state.last_exit_code == 2


def test_missing_log_is_unknown(tmp_path, no_board):
    state = _read(tmp_path, tmp_path / "absent.log")

    assert state.status == fsh.UNKNOWN
    assert state.reason == "no_state"
    assert state.last_exit_code is None
    assert state.last_success_age_seconds is None


# --------------------------------------------------------------------- failing reads


def test_unreadable_log_is_unknown_not_an_exception(tmp_path, no_board):
    unreadable = tmp_path / "a-directory"
    unreadable.mkdir()

    state = _read(tmp_path, unreadable)

    assert state.status == fsh.UNKNOWN
    assert state.reason == "reader_error"


def test_raising_log_reader_is_contained(tmp_path, no_board, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("log reader exploded")

    monkeypatch.setattr(fsh, "scan_fork_sync_log", _boom)

    state = _read(tmp_path, tmp_path / "whatever.log")

    assert state.status == fsh.UNKNOWN
    assert state.reason == "reader_error"


def test_raising_board_reader_is_contained(tmp_path, monkeypatch):
    """A broken board read must not lose the log evidence (nor raise)."""
    log = _write_log(tmp_path, "RESULT created task t_ab12cd (conflict escalated; rc=2)")

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(fsh, "count_open_escalation_tasks", _boom)

    state = _read(tmp_path, log)

    assert state.board_read_failed is True
    assert state.status == fsh.DEGRADED
    assert state.reason == "conflict_escalated"


def test_snapshot_build_survives_a_raising_classifier(tmp_path, monkeypatch):
    """Even a projection bug must not escape into the export loop or the watcher."""
    from agent.monitoring import gateway_health_export as ghe

    def _boom(*args, **kwargs):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(fsh, "read_fork_sync_state", _boom)

    snapshot = fsh.build_fork_sync_health_snapshot(
        fsh.ForkSyncSettings(log_path=tmp_path / "absent.log")
    )

    assert snapshot.state.status == fsh.UNKNOWN
    assert snapshot.state.reason == "reader_error"
    assert {m.name for m in snapshot.metrics} <= set(ghe._OBSERVABLE_METRIC_NAMES)


# --------------------------------------------------------------------------- metrics


def _snapshot(tmp_path, log_path, **kwargs):
    return fsh.build_fork_sync_health_snapshot(
        fsh.ForkSyncSettings(log_path=log_path), now=kwargs.get("now", NOW)
    )


def test_emitted_gauges_are_all_registered(tmp_path, no_board):
    from agent.monitoring import gateway_health_export as ghe

    snapshot = _snapshot(tmp_path, _write_log(tmp_path, "RESULT clean sync (rc=0)"))

    emitted = {m.name for m in snapshot.metrics}
    assert emitted, "the feed must emit gauges"
    assert emitted <= set(ghe._OBSERVABLE_METRIC_NAMES)


def test_metrics_are_content_free(tmp_path, monkeypatch):
    """No task id, path or free-form text may reach a metric value/attribute."""
    log = _write_log(
        tmp_path,
        "RESULT created task t_ab12cd (conflict escalated: 2 files in /opt/data/x; rc=2)",
    )
    db_path = _board_db(tmp_path, [("t_ab12cd", "Fork-sync конфликт: x", "blocked", "k")])
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", _board_reader(db_path))

    snapshot = _snapshot(tmp_path, log)
    blob = " ".join(
        f"{m.name}={m.value!r} {m.attributes!r}" for m in snapshot.metrics
    )

    assert "t_ab12cd" not in blob
    assert "/opt/data" not in blob
    assert "2 files" not in blob  # free-form log text never reaches a metric
    for metric in snapshot.metrics:
        for value in metric.attributes.values():
            assert str(value) in fsh.KNOWN_STATUSES | fsh.KNOWN_REASONS | {"hermes.fork_sync.status", "hermes.fork_sync.reason"}


def test_status_metric_carries_the_bounded_vocabulary(tmp_path, no_board):
    log = _write_log(tmp_path, "RESULT push failed fork/main (rc=3)")

    snapshot = _snapshot(tmp_path, log)
    by_name = {m.name: m for m in snapshot.metrics}

    assert by_name["hermes.fork_sync.up"].value == 0
    assert by_name["hermes.fork_sync.status"].attributes["hermes.fork_sync.status"] == fsh.UNHEALTHY
    assert by_name["hermes.fork_sync.status"].attributes["hermes.fork_sync.reason"] in fsh.KNOWN_REASONS
    assert by_name["hermes.fork_sync.last_exit_code"].value == 3
    assert by_name["hermes.fork_sync.escalations_open"].value == 0


def test_healthy_state_reports_up_and_fresh_ages(tmp_path, no_board):
    log = _write_log(tmp_path, "RESULT clean sync (rc=0)")

    by_name = {m.name: m for m in _snapshot(tmp_path, log).metrics}

    assert by_name["hermes.fork_sync.up"].value == 1
    assert by_name["hermes.fork_sync.last_success_age_seconds"].value == pytest.approx(0.0)
    assert by_name["hermes.fork_sync.last_run_age_seconds"].value == pytest.approx(0.0)


# --------------------------------------------------------------- alerting + caching


def test_transition_into_degraded_logs_one_warning(tmp_path, no_board, monkeypatch, caplog):
    healthy = _write_log(tmp_path, "RESULT clean sync (rc=0)", name="healthy.log")
    failed = _write_log(tmp_path, "RESULT push failed (rc=3)", name="failed.log")

    with caplog.at_level(logging.WARNING, logger=fsh.logger.name):
        fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=healthy), now=NOW)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=failed), now=NOW)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "unhealthy" in warnings[0].getMessage()

        # Same state again: no repeat page.
        fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=failed), now=NOW)
        assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


def test_recovery_logs_info_not_warning(tmp_path, no_board, caplog):
    failed = _write_log(tmp_path, "RESULT push failed (rc=3)", name="failed.log")
    healthy = _write_log(tmp_path, "RESULT clean sync (rc=0)", name="healthy.log")

    with caplog.at_level(logging.INFO, logger=fsh.logger.name):
        fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=failed), now=NOW)
        caplog.clear()
        snapshot = fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=healthy), now=NOW)

    assert snapshot.state.status == fsh.HEALTHY
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("recovered" in r.getMessage() for r in caplog.records)


def test_cached_snapshot_is_served_without_rereading(tmp_path, no_board, monkeypatch):
    log = _write_log(tmp_path, "RESULT clean sync (rc=0)")
    fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=log), now=NOW)

    def _boom(*args, **kwargs):
        raise AssertionError("cached read must not touch the artefacts again")

    monkeypatch.setattr(fsh, "read_fork_sync_state", _boom)

    assert fsh.fork_sync_health_snapshot().state.status == fsh.HEALTHY


def test_cache_expiry_refreshes(tmp_path, no_board, monkeypatch):
    log = _write_log(tmp_path, "RESULT clean sync (rc=0)")
    fsh.refresh_fork_sync_health(fsh.ForkSyncSettings(log_path=log), now=NOW)

    calls = []
    real_read = fsh.read_fork_sync_state

    def _counting(*args, **kwargs):
        calls.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(fsh, "read_fork_sync_state", _counting)
    monkeypatch.setattr(fsh.time, "time", lambda: NOW + fsh._CACHE_TTL_SECONDS + 1.0)

    fsh.fork_sync_health_snapshot()

    assert calls, "an expired cache must re-read the artefacts"


# ------------------------------------------------------------------------- settings


def test_settings_defaults_come_from_config_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["fork_sync"]

    assert fsh.DEFAULT_STALE_AFTER_SECONDS == section["stale_after_seconds"]
    assert fsh.DEFAULT_HEALTH_INTERVAL_SECONDS == section["health_interval_seconds"]
    assert fsh.DEFAULT_TAIL_BYTES == section["tail_bytes"]


def test_settings_honour_config_and_env_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("FORK_SYNC_LOG", raising=False)

    from_config = fsh.resolve_fork_sync_settings({
        "log_path": str(tmp_path / "configured.log"),
        "stale_after_seconds": 7200,
        "board": "factory",
        "health_interval_seconds": 45,
    })

    assert from_config.log_path == tmp_path / "configured.log"
    assert from_config.stale_after_seconds == 7200.0
    assert from_config.board == "factory"
    assert from_config.health_interval_seconds == 45.0

    # The runner's own env var wins over config (single source of truth on disk).
    monkeypatch.setenv("FORK_SYNC_LOG", str(tmp_path / "env.log"))
    assert fsh.resolve_fork_sync_settings({"log_path": "/nope"}).log_path == tmp_path / "env.log"


def test_settings_are_clamped_against_absurd_values(monkeypatch):
    monkeypatch.delenv("FORK_SYNC_LOG", raising=False)

    settings = fsh.resolve_fork_sync_settings({
        "stale_after_seconds": 0, "tail_bytes": 1, "health_interval_seconds": -5,
    })

    assert settings.stale_after_seconds == fsh.DEFAULT_STALE_AFTER_SECONDS
    assert settings.tail_bytes >= 4096
    assert settings.health_interval_seconds >= 30.0


def test_configured_interval_is_clamped_to_the_watcher_floor():
    """The watcher re-clamps the interval, so a tiny configured value can't spin the loop."""
    settings = fsh.resolve_fork_sync_settings({"health_interval_seconds": 1})

    assert settings.health_interval_seconds >= 30.0


def test_state_is_serialisable_for_debugging(tmp_path, no_board):
    log = _write_log(tmp_path, "RESULT clean sync (rc=0)")

    payload = _read(tmp_path, log).to_dict()

    assert payload["status"] == fsh.HEALTHY
    assert "escalation_task_id" not in payload  # ids never leave the projection
    assert time.time() > 0
