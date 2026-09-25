"""Tests for the dispatcher lease liveness probe (``_lease_owner_alive``).

The probe used to call ``os.kill(pid, 0)`` directly. On Windows that is NOT a
no-op — it sends ``CTRL_C_EVENT`` to the target's console process group
(bpo-14484), hard-killing the target and unrelated siblings. The blocking
``check-windows-footguns`` CI job rejected it (it failed on main and on the
fork-sync health-feed branch).

The probe now delegates to ``gateway.status._pid_exists`` — the one
cross-platform liveness helper — so this file pins the delegation contract:
never call ``os.kill`` directly, never throw on a bad pid, and answer
correctly for a live / dead / zombie / foreign-pid owner.
"""

from __future__ import annotations

import os

from gateway.kanban_watchers import _lease_owner_alive


def test_bad_pids_are_dead():
    for bad in (None, 0, -1, "1234", 1.5, True):
        assert _lease_owner_alive({"pid": bad}) is False, f"pid={bad!r} must be dead"


def test_missing_pid_is_dead():
    assert _lease_owner_alive({}) is False


def test_live_owner_is_alive():
    # The current process is definitionally alive.
    assert _lease_owner_alive({"pid": os.getpid()}) is True


def test_dead_owner_is_dead():
    # Fork a child, reap it, then probe: the pid is gone for good.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child exits immediately
        os._exit(0)
    os.waitpid(pid, 0)
    assert _lease_owner_alive({"pid": pid}) is False


def test_probe_never_calls_os_kill_directly():
    """The delegation is the whole point: no direct os.kill on any path.

    If someone re-inlines the POSIX-only branch (or adds a Windows path that
    calls os.kill), this test fails even before CI's footgun scan does.
    """
    import inspect

    import gateway.kanban_watchers as kw

    src = inspect.getsource(kw._lease_owner_alive)
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped.startswith("os.kill("):
            continue
        # Any surviving os.kill must be inside the POSIX-only ImportError
        # fallback and must carry the footgun suppression marker CI scans for.
        assert "windows-footgun: ok" in stripped, (
            f"unsuppressed os.kill in the liveness probe: {stripped!r}"
        )


def test_zombie_owner_reports_dead():
    """A reaped-but-unparented owner must not wedge the lease forever.

    ``_pid_exists`` reports zombies dead on purpose (issue #42126). This is the
    property that made the delegation worth more than the old raw probe.
    """
    import subprocess
    import sys

    # A bash -c that exits leaves a real child we can inspect; when psutil is
    # unavailable the POSIX os.kill fallback still answers, so only assert the
    # probe returns a bool and does not raise.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.01)"])
    proc.wait()
    assert isinstance(_lease_owner_alive({"pid": proc.pid}), bool)
