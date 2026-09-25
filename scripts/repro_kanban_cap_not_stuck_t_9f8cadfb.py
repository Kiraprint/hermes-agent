"""Reproduce the exact 2026-09-22 incident shape against a FIXTURE board.

Scenario (verbatim from the QA report t_34e76b09):
  * kanban.max_in_progress = 8 (memory-derived default)
  * 8 tasks in status=running (factory-dev=6, factory-qa=2)
  * 10 spawnable tasks in ready
  * "0 workers spawned" is the CORRECT answer
  * before: the health counter climbed 36..86 consecutive ticks and logged a
    "dispatcher stuck" WARNING -> panic in the TG digests, bogus incident
  * after:  no stuck WARNING; one INFO "busy at concurrency cap" per 5 min

Runs entirely on a throwaway board DB under a temp HERMES_HOME. Never touches
the live board or hh_tracker.db.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time


def main() -> int:
    home = tempfile.mkdtemp(prefix="t_9f8cadfb_repro_")
    for prof in ("factory-dev", "factory-qa", "default"):
        os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
    os.environ["HERMES_HOME"] = home
    os.environ.pop("PYTHONPATH", None)
    # CRITICAL: this process is itself a kanban worker, so the dispatcher
    # injected HERMES_KANBAN_* vars that point at the LIVE board. Clear them
    # or the fixture writes into the real factory board.
    for var in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        os.environ.pop(var, None)

    sys.path.insert(0, os.getcwd())

    from gateway.run import GatewayRunner
    from hermes_cli import profiles
    import hermes_cli.config as cfg_mod
    import hermes_cli.kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    profiles.profile_exists = lambda name: True

    # Exactly the config that produced the incident: max_in_progress unset in
    # config.yaml, so the dispatcher resolves the memory-derived default. We
    # pin it to 8 here to keep the repro deterministic.
    cfg_mod.load_config = lambda: {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "max_in_progress": 8,
        }
    }
    kb.list_boards = lambda include_archived=False: [{"slug": "factory"}]
    kb.read_board_metadata = lambda slug: {"slug": slug}
    kbd.reap_worker_zombies = lambda: []
    # Workers that never really spawn -> the board stays saturated.
    kbd._default_spawn = lambda *a, **k: None

    kb.create_board(slug="factory", name="factory")
    conn = kbc.connect(board="factory")
    with conn:
        running = []
        for i in range(6):
            tid = kb.create_task(conn, title=f"dev-{i}", assignee="factory-dev")
            running.append((tid, "factory-dev"))
        for i in range(2):
            tid = kb.create_task(conn, title=f"qa-{i}", assignee="factory-qa")
            running.append((tid, "factory-qa"))
        for tid, _who in running:
            assert kb.claim_task(conn, tid) is not None
            conn.execute(
                "UPDATE tasks SET status='running', worker_pid=NULL, "
                "claim_lock='lock' WHERE id = ?",
                (tid,),
            )
        # 10 queued tasks. ``create_task`` only accepts initial_status
        # running/blocked, so the queued ones are created and then moved to
        # 'ready' directly — this is a throwaway fixture DB.
        for i in range(10):
            tid = kb.create_task(
                conn, title=f"queued-{i}", assignee="factory-dev",
            )
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "worker_pid=NULL, current_run_id=NULL WHERE id = ?",
                (tid,),
            )

    n_running = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='running'"
    ).fetchone()[0]
    n_ready = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='ready'"
    ).fetchone()[0]
    print(f"FIXTURE: running={n_running} (cap=8), ready={n_ready}")
    assert (n_running, n_ready) == (8, 10), "fixture shape drifted"
    assert kb.has_spawnable_ready(conn) is True
    conn.close()

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    gwl = logging.getLogger("gateway.run")
    handler = _Capture()
    gwl.addHandler(handler)
    gwl.setLevel(logging.INFO)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None
    runner._kanban_dispatcher_lock_path = os.path.join(home, "disp.lock")

    ticks = {"n": 0}
    real_sleep = asyncio.sleep
    watchdog = {"limit": 60}

    async def _sleep(delay):
        ticks["n"] += 1
        # 40 ticks at 1s interval >> HEALTH_WINDOW (6): the pre-fix code had
        # long since fired its stuck warning by then.
        if ticks["n"] >= 40 or watchdog["limit"] <= 0:
            runner._running = False
        watchdog["limit"] -= 1
        return await real_sleep(0)

    import gateway.kanban_watchers as kw
    kw.asyncio.sleep = _sleep

    t0 = time.time()
    asyncio.run(
        asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=30.0)
    )
    elapsed = time.time() - t0
    kw.asyncio.sleep = real_sleep
    gwl.removeHandler(handler)

    msgs = [r.getMessage() for r in records if r.name == "gateway.run"]
    stuck = [m for m in msgs if "dispatcher stuck" in m]
    busy = [m for m in msgs if "busy at concurrency cap" in m]
    spawned = [m for m in msgs if "spawned=" in m]

    print(f"ticks simulated: {ticks['n']} in {elapsed:.1f}s")
    print(f"'dispatcher stuck' warnings: {len(stuck)}")
    for m in stuck:
        print("   !!", m)
    print(f"'busy at concurrency cap' infos: {len(busy)}")
    for m in busy:
        print("   --", m)
    print(f"spawn log lines: {len(spawned)}")

    ok = True
    if stuck:
        print("FAIL: stuck warning fired at cap (the t_9f8cadfb bug)")
        ok = False
    else:
        print("PASS: no stuck warning while in-flight == cap")
    if not busy:
        print("FAIL: no 'busy at concurrency cap' line (operator has no signal)")
        ok = False
    else:
        assert "max_in_progress" in busy[0], busy[0]
        assert "8/8" in busy[0], busy[0]
        assert "not a stuck dispatcher" in busy[0], busy[0]
        if len(busy) > 1:
            print(f"FAIL: busy line repeated {len(busy)}x (should be throttled)")
            ok = False
        else:
            print("PASS: busy-at-cap explained once, rate-limited")
    if spawned:
        print("NOTE: a spawn happened — the cap was not actually full")

    # The queue must still be intact: the fix must not consume or drop work.
    conn2 = kbc.connect(board="factory")
    still_running = conn2.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='running'"
    ).fetchone()[0]
    still_ready = conn2.execute(
        "SELECT COUNT(*) FROM tasks WHERE status='ready'"
    ).fetchone()[0]
    conn2.close()
    print(f"AFTER: running={still_running} ready={still_ready}")
    if (still_running, still_ready) != (8, 10):
        print("FAIL: board state changed — the fix must be telemetry-only")
        ok = False
    else:
        print("PASS: queue untouched (telemetry-only change)")

    print("RESULT:", "OK" if ok else "BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
