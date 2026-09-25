"""Regression tests for t_9f8cadfb — "cap reached" must not read as "stuck".

The dispatcher health telemetry used to count a tick as *bad* whenever a
ready+spawnable task existed and nothing spawned. During 2026-09-22
03:09-03:59 UTC that fired for 36-86 consecutive ticks on a perfectly healthy
factory board: ``max_in_progress=8`` was fully saturated (8 running, 10 in
ready) and ``0 workers spawned`` was the CORRECT answer. The resulting
"dispatcher stuck" WARNING caused panic in the Telegram digests and seeded a
bogus incident (t_5d692058).

A concurrency cap is a normal steady state — the backlog drains as running
tasks finish. These tests pin:
  1. global ``max_in_progress`` at cap -> NOT stuck;
  2. per-board ``max_spawn`` at cap -> NOT stuck;
  3. per-profile ``max_in_progress_per_profile`` at cap -> NOT stuck;
  4. headroom + zero spawns -> still stuck (the real signal survives);
  5. a genuine fault behind a saturated board is still reported;
  6. a cap-blocked tick clears itself when a worker finishes.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def board(monkeypatch):
    """Fresh HERMES_HOME with a board and real profile dirs alpha/beta."""
    test_home = tempfile.mkdtemp(prefix="kanban_cap_not_stuck_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state"):
            del sys.modules[mod]
    from hermes_cli import kanban_db

    yield kanban_db
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state"):
            del sys.modules[mod]


def _fake_spawn(*args, **kwargs):
    return 4242


def _park_running(conn, task_id):
    """Move a claimed task into 'running' without a live worker."""
    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'running', worker_pid = NULL, "
            "claim_lock = 'lock' WHERE id = ?",
            (task_id,),
        )


def _add_tasks(conn, count, assignee="alpha"):
    from hermes_cli import kanban_db as kb

    return [
        kb.create_task(conn, title=f"t{i}", assignee=assignee)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 1. global max_in_progress reached -> "busy", not "stuck"
# ---------------------------------------------------------------------------


def test_global_cap_zero_spawn_is_not_stuck(board):
    """in-flight == max_in_progress with a non-empty ready queue is busy.

    This is the exact shape from the 2026-09-22 incident: 8 running against
    a cap of 8, 10 spawnable tasks waiting, 0 spawned.
    """
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        board.create_board(slug="default", name="T")
        ids = _add_tasks(conn, 10)
        # Saturate: claim 8 and park them in 'running'.
        for tid in ids[:8]:
            assert board.claim_task(conn, tid) is not None
            _park_running(conn, tid)
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=8,
        )

    assert res.spawned == []
    assert res.cap_reason == "max_in_progress"
    assert res.cap_running == 8
    assert res.cap_limit == 8
    assert kbd.dispatch_cap_busy(res) is True


def test_cap_busy_flag_clears_once_a_worker_finishes(board):
    """The cap is per-tick state: completing one task unblocks a spawn."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        board.create_board(slug="default", name="T")
        ids = _add_tasks(conn, 10)
        for tid in ids[:8]:
            assert board.claim_task(conn, tid) is not None
            _park_running(conn, tid)
        capped = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress=8)
        assert kbd.dispatch_cap_busy(capped) is True

        # One worker finishes -> headroom of 1.
        with board.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (ids[0],),
            )
        freed = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress=8)

    assert len(freed.spawned) == 1
    assert freed.cap_reason is None
    assert kbd.dispatch_cap_busy(freed) is False


# ---------------------------------------------------------------------------
# 2. per-board max_spawn reached
# ---------------------------------------------------------------------------


def test_max_spawn_cap_zero_spawn_is_not_stuck(board):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        board.create_board(slug="default", name="T")
        ids = _add_tasks(conn, 5)
        for tid in ids[:2]:
            assert board.claim_task(conn, tid) is not None
            _park_running(conn, tid)
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_spawn=2)

    assert res.spawned == []
    assert res.cap_reason == "max_spawn"
    assert kbd.dispatch_cap_busy(res) is True


# ---------------------------------------------------------------------------
# 3. per-profile cap reached
# ---------------------------------------------------------------------------


def test_per_profile_cap_zero_spawn_is_not_stuck(board):
    """Every ready task belongs to a profile already at its own cap."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        board.create_board(slug="default", name="T")
        ids = _add_tasks(conn, 3, assignee="alpha")
        assert board.claim_task(conn, ids[0]) is not None
        _park_running(conn, ids[0])
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress_per_profile=1,
        )

    assert res.spawned == []
    assert res.skipped_per_profile_capped
    assert res.cap_reason == "max_in_progress_per_profile"
    assert kbd.dispatch_cap_busy(res) is True


# ---------------------------------------------------------------------------
# 4. headroom + zero spawns -> the real stuck signal must survive
# ---------------------------------------------------------------------------


def test_uncapped_zero_spawn_is_still_stuck(board):
    """No cap in play and nothing spawned: ready work is being refused.

    This is the broken-PATH / missing-venv / credential-loss case the warning
    exists for. It must NOT be suppressed. The spawn function raising models a
    worker that cannot start at all (bad venv / PATH), with a high
    ``failure_limit`` so the task is released back to ready rather than
    auto-blocked.
    """
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_connect as kbc

    def _broken_spawn(*args, **kwargs):
        raise RuntimeError("No module named hermes (broken venv)")

    with kbc.connect_closing() as conn:
        board.create_board(slug="default", name="T")
        _add_tasks(conn, 3)
        # max_in_progress well above in-flight -> budget allows spawning.
        res = kbd.dispatch_once(
            conn, spawn_fn=_broken_spawn, max_in_progress=8, failure_limit=99,
        )

    assert res.spawned == []
    assert res.cap_reason is None
    assert kbd.dispatch_cap_busy(res) is False


# ---------------------------------------------------------------------------
# 5. a real fault behind a saturated board still surfaces
# ---------------------------------------------------------------------------


def test_spawn_failure_behind_saturated_board_is_not_masked(board):
    """Cap-busy detection must not swallow a genuine spawn fault.

    Conservative by construction: a tick that also auto-blocked / rate-limited
    / respawn-guarded a task is not reported as cap-busy even when the cap is
    also full, so health telemetry still raises the stuck warning.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    saturated = kbd.DispatchResult(
        cap_reason="max_in_progress", cap_running=8, cap_limit=8,
    )
    saturated.skipped_per_profile_capped = [("t1", "alpha", 1)]
    saturated.auto_blocked = ["t2"]
    assert kbd.dispatch_cap_busy(saturated) is False

    rate_limited = kbd.DispatchResult(cap_reason="max_in_progress")
    rate_limited.skipped_per_profile_capped = [("t1", "alpha", 1)]
    rate_limited.rate_limited = ["t2"]
    assert kbd.dispatch_cap_busy(rate_limited) is False

    respawn_guarded = kbd.DispatchResult(cap_reason="max_in_progress")
    respawn_guarded.skipped_per_profile_capped = [("t1", "alpha", 1)]
    respawn_guarded.respawn_guarded = [("t2", "active_pr")]
    assert kbd.dispatch_cap_busy(respawn_guarded) is False

    unassigned = kbd.DispatchResult(cap_reason="max_in_progress")
    unassigned.skipped_per_profile_capped = [("t1", "alpha", 1)]
    unassigned.skipped_unassigned = ["t2"]
    assert kbd.dispatch_cap_busy(unassigned) is False

    # A tick that did spawn is never "cap busy" regardless of other buckets.
    spawned = kbd.DispatchResult(cap_reason="max_in_progress")
    spawned.spawned = [("t1", "alpha", "/tmp/ws")]
    assert kbd.dispatch_cap_busy(spawned) is False

    # Pure per-profile deferral with no other bucket -> busy, not stuck.
    clean = kbd.DispatchResult()
    clean.skipped_per_profile_capped = [("t1", "alpha", 1)]
    assert kbd.dispatch_cap_busy(clean) is True


# ---------------------------------------------------------------------------
# 6. gateway-level: the warning itself
# ---------------------------------------------------------------------------


def test_gateway_dispatcher_emits_no_stuck_warning_at_cap(
    monkeypatch, tmp_path, caplog
):
    """End-to-end: saturate the cap, run real dispatcher ticks, assert the
    "dispatcher stuck" WARNING never appears (only the INFO busy line)."""
    import asyncio
    import logging

    from gateway.run import GatewayRunner
    from hermes_cli import profiles as _profiles

    test_home = tmp_path / "home"
    for prof in ("alpha", "beta", "default"):
        (test_home / "profiles" / prof).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state"):
            del sys.modules[mod]
    # Import AFTER the sys.modules purge so the patches below land on the
    # exact module objects the watcher will import at runtime.
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_connect as _kbc
    from hermes_cli import kanban_db_dispatch as _kbd
    monkeypatch.setattr(_profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        _cfg_mod, "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "max_in_progress": 2,
                "spawn_fn": None,
            }
        },
    )
    monkeypatch.setattr(_kb, "list_boards", lambda include_archived=False: [{"slug": "default"}])
    monkeypatch.setattr(_kb, "read_board_metadata", lambda slug: {"slug": slug})

    _kb.create_board(slug="default", name="T")
    with _kbc.connect_closing() as conn:
        ids = _add_tasks(conn, 8)
        for tid in ids[:2]:
            assert _kb.claim_task(conn, tid) is not None
            _park_running(conn, tid)

    # No real workers are spawned (spawn_fn -> None), so the board stays at cap.
    monkeypatch.setattr(_kbd, "_default_spawn", lambda *a, **k: None)
    monkeypatch.setattr(_kbd, "reap_worker_zombies", lambda: [])

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None
    runner._kanban_dispatcher_lock_path = str(tmp_path / "disp.lock")

    ticks = {"n": 0}
    real_sleep = asyncio.sleep

    async def _sleep(delay):
        ticks["n"] += 1
        if ticks["n"] >= 20:  # 20 ticks > HEALTH_WINDOW (6)
            runner._running = False
        return await real_sleep(0)

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_kanban_dispatcher_lock",
        lambda *a, **k: True,
        raising=False,
    )

    caplog.set_level(logging.DEBUG, logger="gateway.run")
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        asyncio.run(runner._kanban_dispatcher_watcher())

    records = [r.getMessage() for r in caplog.records]
    stuck = [m for m in records if "dispatcher stuck" in m]
    assert stuck == [], f"cap-reached must not warn 'stuck', got: {stuck}"
    busy = [m for m in records if "busy at concurrency cap" in m]
    assert busy, f"expected an INFO 'busy at concurrency cap' line, got: {records}"
    assert "not a stuck dispatcher" in busy[0]
    assert "max_in_progress" in busy[0]

    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state"):
            del sys.modules[mod]
