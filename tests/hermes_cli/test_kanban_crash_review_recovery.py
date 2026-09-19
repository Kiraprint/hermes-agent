"""Crash-requeue review-handoff recovery (mechanism A, incident t_91c7c52f).

A worker opens a PR, posts the PR URL as a task comment, then crashes before
calling ``kanban_complete`` / ``request_review`` (rc=0 protocol violation, or
pid-not-alive). The reclaim paths requeue the task to ``ready`` — where the
READY-lane ``active_pr`` respawn guard parks it for up to 24h
(_RESPAWN_GUARD_PR_WINDOW). The board stalled 22.7h on exactly this.

Fix: ``_crash_recovery_status`` redirects such requeues into the review lane,
which already skips ``active_pr``/``recent_success`` in
``check_respawn_guard``. Guards are NOT weakened: tasks without a fresh PR
comment behave exactly as before.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "PR: https://github.com/acme/widgets/pull/42 — ready for review"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _dead_pid(raw_status: int | None = None) -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    if raw_status is not None:
        # Seed the reap registry so _classify_worker_exit sees this exit kind
        # (raw_status=0 → "clean_exit" → rc=0 protocol violation).
        kbd._record_worker_exit(p.pid, raw_status)
    return p.pid


def _crashed_task_with_pr(conn, tid: str) -> None:
    """Drive a task to running-with-dead-pid + fresh PR comment."""
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid())
    kb.add_comment(conn, tid, "worker", PR_COMMENT)


# ---------------------------------------------------------------------------
# Scenario A: rc=0 protocol-violation path (detect_crashed_workers)
# ---------------------------------------------------------------------------


def test_protocol_violation_with_fresh_pr_recovers_to_review(conn):
    tid = kb.create_task(conn, title="ship", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid(raw_status=0))
    kb.add_comment(conn, tid, "worker", PR_COMMENT)

    crashed = kbd.detect_crashed_workers(conn)
    assert tid in crashed
    task = kb.get_task(conn, tid)
    assert task.status == "review", (
        "crash-requeue of a task with a fresh PR comment must land in the "
        "review lane, not ready under the active_pr guard"
    )

    # The crash event records the auto-recovery with the PR reference.
    events = [
        e for e in kb.list_events(conn, tid) if e.kind == "protocol_violation"
    ]
    assert events
    recovery = events[-1].payload.get("crash_review_recovery")
    assert recovery == {
        "auto": True,
        "pr_url": "https://github.com/acme/widgets/pull/42",
        "from": "ready",
        "to": "review",
    }


def test_recovered_task_spawns_from_review_lane(conn, monkeypatch):
    """End-to-end guard check: the recovered card must NOT be held by the
    active_pr respawn guard when dispatching from the review lane."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)

    tid = kb.create_task(conn, title="ship", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid(raw_status=0))
    kb.add_comment(conn, tid, "worker", PR_COMMENT)
    kbd.detect_crashed_workers(conn)
    assert kb.get_task(conn, tid).status == "review"

    # The READY lane would defer under active_pr; the review lane allows it.
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"
    assert kbd.check_respawn_guard(conn, tid, lane="review") is None


# ---------------------------------------------------------------------------
# Scenario B: pid-not-alive path (detect_crashed_workers)
# ---------------------------------------------------------------------------


def test_dead_pid_with_fresh_pr_recovers_to_review(conn):
    tid = kb.create_task(conn, title="ship", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid())
    # No reap-registry entry → kind == "unknown" → plain crashed event,
    # which is the real-world "pid … not alive" detection path.
    kb.add_comment(conn, tid, "worker", PR_COMMENT)

    crashed = kbd.detect_crashed_workers(conn)
    assert tid in crashed
    assert kb.get_task(conn, tid).status == "review"

    events = [e for e in kb.list_events(conn, tid) if e.kind == "crashed"]
    assert events
    recovery = events[-1].payload.get("crash_review_recovery")
    assert recovery and recovery["pr_url"].endswith("/pull/42")


# ---------------------------------------------------------------------------
# Regression: no PR comment → behaviour unchanged, guards intact
# ---------------------------------------------------------------------------


def test_no_pr_comment_requeues_ready_as_before(conn):
    tid = kb.create_task(conn, title="plain", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid())

    crashed = kbd.detect_crashed_workers(conn)
    assert tid in crashed
    task = kb.get_task(conn, tid)
    assert task.status == "ready", (
        "tasks without PR comments must keep the legacy requeue behaviour"
    )
    assert all(
        e.payload.get("crash_review_recovery") is None
        for e in kb.list_events(conn, tid)
        if e.kind in ("crashed", "protocol_violation")
    )


def test_stale_pr_comment_does_not_recover(conn, monkeypatch):
    tid = kb.create_task(conn, title="stale pr", assignee="w")
    kb.claim_task(conn, tid)
    pid = _dead_pid()
    kbd._set_worker_pid(conn, tid, pid)

    old = int(__import__("time").time()) - kbd._RESPAWN_GUARD_PR_WINDOW - 60
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (tid, "worker", PR_COMMENT, old),
    )
    conn.commit()

    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kb.get_task(conn, tid).status == "ready", (
        "a PR older than the respawn-guard window must NOT trigger recovery"
    )


def test_kill_switch_disables_recovery(conn, monkeypatch):
    monkeypatch.setenv("KANBAN_CRASH_REVIEW_RECOVERY", "0")
    tid = kb.create_task(conn, title="switched off", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid())
    kb.add_comment(conn, tid, "worker", PR_COMMENT)

    kbd.detect_crashed_workers(conn)
    assert kb.get_task(conn, tid).status == "ready"


def test_latest_pr_url_wins_in_summary_payload(conn):
    tid = kb.create_task(conn, title="two prs", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid(raw_status=0))
    kb.add_comment(conn, tid, "worker", PR_COMMENT)
    newer = (
        "superseded by https://github.com/acme/widgets/pull/99 (force-push)"
    )
    kb.add_comment(conn, tid, "worker", newer)

    kbd.detect_crashed_workers(conn)
    events = [
        e for e in kb.list_events(conn, tid) if e.kind == "protocol_violation"
    ]
    recovery = events[-1].payload.get("crash_review_recovery")
    assert recovery["pr_url"] == "https://github.com/acme/widgets/pull/99"


# ---------------------------------------------------------------------------
# release_stale_claims path (TTL-expired claim instead of dead pid)
# ---------------------------------------------------------------------------


def test_stale_claim_with_fresh_pr_recovers_to_review(conn):
    tid = kb.create_task(conn, title="ttl stale", assignee="w")
    kb.claim_task(conn, tid)
    # Force the claim to be TTL-expired (claim_task clamps ttl_seconds >= 1).
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(__import__("time").time()) - 1, tid),
    )
    conn.commit()
    kb.add_comment(conn, tid, "worker", PR_COMMENT)

    reclaimed = kb.release_stale_claims(conn)
    assert reclaimed == 1
    assert kb.get_task(conn, tid).status == "review"


def test_stale_claim_without_pr_reclaims_ready(conn):
    tid = kb.create_task(conn, title="ttl stale plain", assignee="w")
    kb.claim_task(conn, tid)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(__import__("time").time()) - 1, tid),
    )
    conn.commit()

    reclaimed = kb.release_stale_claims(conn)
    assert reclaimed == 1
    assert kb.get_task(conn, tid).status == "ready"


def test_stale_comment_not_guarded(conn):
    """Expired PR comment must not hold the task under active_pr.

    Regression: ``check_respawn_guard`` uses ``created_at >= pr_cutoff``,
    so a comment older than ``_RESPAWN_GUARD_PR_WINDOW`` must return None
    even when a fresh PR URL is present.
    """
    tid = kb.create_task(conn, title="stale comment", assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, _dead_pid(raw_status=0))
    old = int(__import__("time").time()) - kbd._RESPAWN_GUARD_PR_WINDOW - 60
    kb.add_comment(conn, tid, "worker", PR_COMMENT, created_at=old)

    assert kbd.check_respawn_guard(conn, tid) is None
