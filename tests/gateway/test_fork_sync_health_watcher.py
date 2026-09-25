"""Tests for the fork-sync health watcher, readiness probe and dispatch invariant.

The feed is observational by contract: it reports the nightly fork/upstream
sync's liveness, and a failed feed must never stall the dispatcher, the ready
queue or the readiness endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.monitoring import fork_sync_health as fsh
from gateway import kanban_watchers
from gateway import readiness
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

NOW = 1_800_000_000.0


class _Runner(GatewayKanbanWatchersMixin):
    """Minimal stand-in for GatewayRunner: the watcher only needs ``_running``."""

    def __init__(self):
        self._running = True


def _log(tmp_path, *messages, name="fork-sync.log"):
    path = tmp_path / name
    stamp = datetime.fromtimestamp(NOW, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text("".join(f"[{stamp}] {msg}\n" for msg in messages), encoding="utf-8")
    return path


def _settings(log_path, **kwargs):
    return fsh.ForkSyncSettings(log_path=log_path, **kwargs)


@pytest.fixture(autouse=True)
def _clean_cache():
    fsh.reset_fork_sync_health_cache()
    yield
    fsh.reset_fork_sync_health_cache()


@pytest.fixture()
def one_tick(monkeypatch):
    """Stop the watcher loop after exactly one iteration (no real sleeping)."""
    import gateway.run_watchers as run_watchers

    async def _stop(runner, seconds):
        runner._running = False

    monkeypatch.setattr(run_watchers, "_interruptible_sleep", _stop)
    monkeypatch.setattr(kanban_watchers, "_FORK_SYNC_HEALTH_SETTLE_SECONDS", 0.0)


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    """A private HERMES_HOME/kanban home so dispatch tests never touch a real board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture()
def spawnable_assignees(monkeypatch):
    """Synthetic assignees must pass the dispatcher's profile-exists guard."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


# ----------------------------------------------------------------- registration


def test_watcher_is_registered_for_gateway_startup():
    from gateway.run import GatewayRunner

    assert "_fork_sync_health_watcher" in GatewayRunner._PRE_RECONNECT_WATCHERS
    assert callable(getattr(GatewayRunner, "_fork_sync_health_watcher", None))


# ------------------------------------------------------------------- loop behaviour


def test_watcher_records_the_feed_state(one_tick, tmp_path, monkeypatch):
    monkeypatch.setattr(
        fsh, "resolve_fork_sync_settings",
        lambda section=None: _settings(_log(tmp_path, "RESULT push failed fork/main (rc=3)")),
    )
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", lambda board=None, **kw: 0)
    runner = _Runner()

    asyncio.run(runner._fork_sync_health_watcher())

    assert runner._fork_sync_health.status == fsh.UNHEALTHY
    assert runner._fork_sync_health.reason == "push_failed"


def test_watcher_survives_a_raising_refresh(one_tick, monkeypatch, caplog):
    """A projection bug is logged, never propagated — the loop must keep ticking."""
    def _boom(*args, **kwargs):
        raise RuntimeError("refresh exploded")

    monkeypatch.setattr(
        fsh, "resolve_fork_sync_settings", lambda section=None: _settings(None)
    )
    monkeypatch.setattr(fsh, "refresh_fork_sync_health", _boom)
    runner = _Runner()

    with caplog.at_level(logging.WARNING, logger=kanban_watchers.logger.name):
        asyncio.run(runner._fork_sync_health_watcher())  # must not raise

    assert any("error_type=RuntimeError" in r.getMessage() for r in caplog.records)
    assert runner._running is False  # exited through the loop, not through an exception


def test_watcher_returns_immediately_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fsh, "resolve_fork_sync_settings",
        lambda section=None: _settings(None, enabled=False),
    )
    monkeypatch.setattr(fsh, "refresh_fork_sync_health", lambda *a, **k: calls.append(1))
    runner = _Runner()

    asyncio.run(runner._fork_sync_health_watcher())

    assert calls == []
    assert runner._running is True  # never entered the loop


# -------------------------------------------------------------- dispatch invariant


def test_failed_fork_sync_does_not_stop_dispatch(
    kanban_home, spawnable_assignees, tmp_path, monkeypatch,
):
    """Invariant (ticket): with fork-sync failed, the dispatcher still runs and the
    ready queue stays non-empty — no deadlock, no freeze."""
    log = _log(tmp_path, "RESULT push failed fork/main -> deadbee (rc=3)")

    def _boom(*args, **kwargs):
        raise RuntimeError("board read exploded")

    # fork-sync is failed hard: last run failed AND the board read is broken.
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", _boom)
    snapshot = fsh.refresh_fork_sync_health(_settings(log), now=NOW)
    assert snapshot.state.status == fsh.UNHEALTHY

    with kbc.connect() as conn:
        kb.create_task(conn, title="ready work A", assignee="worker", board="default")
        kb.create_task(conn, title="ready work B", assignee="worker", board="default")

        spawned = []
        result = kbd.dispatch_once(
            conn, board="default", max_spawn=1,
            spawn_fn=lambda *a, **k: spawned.append(a) or 12345,
        )

        assert spawned, f"dispatcher spawned nothing while fork-sync was failed: {result}"
        still_ready = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status = 'ready'"
        ).fetchone()["n"]

    assert still_ready >= 1, "the ready queue must stay serviced and non-empty"
    # And the feed is still reporting the failure (dispatch did not mask or reset it).
    assert fsh.fork_sync_health_snapshot().state.status == fsh.UNHEALTHY


# ------------------------------------------------------------------- readiness


def _snapshot(status, reason, **kwargs):
    return fsh.ForkSyncHealthSnapshot(
        state=fsh.ForkSyncState(status=status, reason=reason, **kwargs), metrics=[]
    )


def _readiness():
    return readiness.collect_runtime_readiness(
        configured_model="test/model", runtime_status={"gateway_state": "running"},
    )


def test_readiness_reports_degraded_fork_sync(monkeypatch):
    monkeypatch.setattr(
        fsh, "fork_sync_health_snapshot",
        lambda **kw: _snapshot(fsh.DEGRADED, "stale_sync", last_success_age_seconds=200_000.0),
    )

    result = _readiness()

    check = result["checks"]["fork_sync"]
    assert check["status"] == "degraded"
    assert check["state"] == fsh.DEGRADED
    assert check["reason"] == "stale_sync"
    # Content-free invariant: readiness never exposes paths or free-form text.
    assert "/" not in json.dumps(check)


def test_readiness_keeps_an_unknown_fork_sync_ok(monkeypatch):
    """A host that never ran the nightly runner is not 'degraded'."""
    monkeypatch.setattr(
        fsh, "fork_sync_health_snapshot", lambda **kw: _snapshot(fsh.UNKNOWN, "no_state"),
    )

    check = _readiness()["checks"]["fork_sync"]

    assert check["status"] == "ok"
    assert check["state"] == fsh.UNKNOWN


def test_readiness_fork_sync_probe_never_raises(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(fsh, "fork_sync_health_snapshot", _boom)

    check = _readiness()["checks"]["fork_sync"]

    assert check["status"] == "ok"
    assert "unavailable" in check["detail"]


def test_readiness_uses_the_cached_snapshot(tmp_path, monkeypatch):
    """The probe must share the watcher's cache, not re-read the artefacts per poll."""
    monkeypatch.setattr(
        fsh, "resolve_fork_sync_settings",
        lambda section=None: _settings(_log(tmp_path, "RESULT clean sync (rc=0)")),
    )
    monkeypatch.setattr(fsh, "count_open_escalation_tasks", lambda board=None, **kw: 0)
    fsh.refresh_fork_sync_health(_settings(_log(tmp_path, "RESULT clean sync (rc=0)")), now=NOW)

    def _boom(*args, **kwargs):
        raise AssertionError("readiness must use the cached snapshot")

    monkeypatch.setattr(fsh, "read_fork_sync_state", _boom)

    check = _readiness()["checks"]["fork_sync"]

    assert check["status"] == "ok"
    assert check["state"] == fsh.HEALTHY
