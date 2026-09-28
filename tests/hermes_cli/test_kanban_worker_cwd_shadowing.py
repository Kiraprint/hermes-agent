from __future__ import annotations
import subprocess
import sys


def test_module_hermes_argv_keeps_cwd_off_sys_path():
    """The worker harness must not be importable from the task workspace.

    Dispatcher-spawned workers run with TERMINAL_CWD set to their task
    workspace, and for repo-patching tasks that workspace is a checkout of this
    very repository. ``python -m`` prepends the cwd to ``sys.path``, so without
    ``-P`` the worker imports ``tools.*``/``agent.*`` from that (possibly stale)
    checkout instead of the deployed image. Observed 2026-09-28: a Sep-22
    workspace copy of ``tools/daemon_pool.py`` lacked the Python-3.14
    worker-context branch, so every tool call failed with
    ``'DaemonThreadPoolExecutor' object has no attribute '_initializer'`` and
    the card crash-looped into an auto-block.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    argv = kbd._module_hermes_argv()
    assert argv[0] == sys.executable, argv
    assert "-P" in argv, argv
    assert argv.index("-P") < argv.index("-m"), argv
    assert argv[argv.index("-m") + 1] == "hermes_cli.main", argv


def test_dash_p_drops_cwd_from_sys_path(tmp_path):
    """Mechanism behind the flag, exercised on the real interpreter: with -P the
    current directory is not on sys.path (so a workspace copy cannot shadow the
    installed package); without it, it is."""
    probe = "import sys; print([p for p in sys.path if p in ('', '.')])"

    shadowed = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    safe = subprocess.run(
        [sys.executable, "-P", "-c", probe], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )

    assert shadowed.stdout.strip() == "['']"
    assert safe.stdout.strip() == "[]"
