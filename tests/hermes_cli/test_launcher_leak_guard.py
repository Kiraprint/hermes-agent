"""A task workspace must not own a shared PATH entry.

A kanban worker boots Hermes from its own scratch workspace, and boot
maintenance used to publish the PATH conveniences *from that checkout*::

    $HOME/.local/bin/hermes -> .../kanban/workspaces/t_7180bf50/.hermes/bin/hermes

The workspace is deleted when the task completes, so ``hermes`` kept resolving
to a dangling shim for every later session that has that bin dir on PATH —
exactly what broke the factory dispatcher (t_7180bf50, t_a1a9f616, t_157e9c96).

The barrier has two halves, pinned here: publication from an ephemeral checkout
is refused fail-closed (``hermes_cli.launcher_leak_guard`` sits in front of
``_launchers.expose_cli``), and worker teardown sweeps whatever an older build
already left in the shared bin dirs.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import _launchers
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import launcher_leak_guard as guard


def test_publication_from_a_task_workspace_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A workspace checkout is never allowed to claim the shared command."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Keep the sweep's blast radius inside the test.
    monkeypatch.setenv(guard.SHARED_BIN_DIRS_ENV, str(home / ".local" / "bin"))

    workspaces = tmp_path / "kanban" / "boards" / "factory" / "workspaces"
    checkout = workspaces / "t_157e9c96" / "hermes-wt"
    checkout.mkdir(parents=True)
    durable = tmp_path / "opt" / "hermes"
    durable.mkdir(parents=True)

    assert guard.is_ephemeral_root(checkout) is True
    assert guard.is_ephemeral_root(durable) is False

    result = _launchers.expose_cli(project_root=checkout, create=False)
    assert result["ok"] is True
    assert result.get("skipped") == "ephemeral-root"
    assert not result.get("written")
    assert not (home / ".local" / "bin").exists(), "workspace checkout published a shared launcher"

    # The same call from a durable checkout is not refused: the guard keys on
    # the ephemeral root, not on "no publication ever".
    control = _launchers.expose_cli(project_root=durable, create=False)
    assert control.get("skipped") != "ephemeral-root"


def test_scrub_removes_workspace_entries_only(tmp_path: Path) -> None:
    """The sweep clears workspace-pointing files and leaves everything else."""
    workspaces = tmp_path / "kanban" / "boards" / "factory" / "workspaces"
    leaked = workspaces / "t_7180bf50"
    binn = tmp_path / "bin"
    binn.mkdir()

    wrapper = binn / "hermes"
    wrapper.write_text(
        f'#!/bin/sh\nexec {leaked}/.hermes/bin/hermes --run-module hermes_cli.main "$@"\n',
        encoding="utf-8",
    )
    acp = binn / "hermes-acp"
    acp.symlink_to(leaked / ".hermes" / "bin" / "hermes-acp")  # dangling: ws is gone
    stale = binn / "hermes-agent"
    stale.write_text(f"#!/bin/sh\nexec {leaked}/.hermes/bin/hermes-agent \"$@\"\n", encoding="utf-8")
    foreign = binn / "hermes-doctor-helper"
    foreign.symlink_to("/usr/bin/true")
    durable = binn / "other-tool"
    durable.write_text("#!/bin/sh\nexec /opt/hermes/.venv/bin/hermes doctor \"$@\"\n", encoding="utf-8")

    assert guard.leak_reason(wrapper, [workspaces]) is not None
    assert guard.leak_reason(foreign, [workspaces]) is None
    assert guard.leak_reason(durable, [workspaces]) is None

    dry = guard.scrub_shared_bin_leaks(roots=[workspaces], dirs=[binn], dry_run=True)
    removed_names = {Path(entry.split(" (")[0]).name for entry in dry["removed"]}
    assert removed_names == {"hermes", "hermes-acp", "hermes-agent"}
    assert wrapper.exists(), "dry run must not touch the filesystem"

    report = guard.scrub_shared_bin_leaks(roots=[workspaces], dirs=[binn])
    removed_names = {Path(entry.split(" (")[0]).name for entry in report["removed"]}
    assert removed_names == {"hermes", "hermes-acp", "hermes-agent"}
    assert not wrapper.exists() and not acp.exists() and not stale.exists()
    assert foreign.is_symlink() and durable.exists(), "sweep ate an unrelated launcher"
    assert report["errors"] == []

    # A scrubbed `hermes` can be repointed at a durable CLI instead of vanishing.
    repaired = guard.scrub_shared_bin_leaks(
        roots=[workspaces], dirs=[binn], repair_hermes="/opt/hermes/.venv/bin/hermes"
    )
    assert repaired["removed"] == []  # nothing leaky is left
    assert binn.joinpath("hermes-agent").exists() is False


def test_task_teardown_sweeps_the_shim_it_left(tmp_path: Path, monkeypatch) -> None:
    """Completing/cleaning up a task removes the shared shim it leaked."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    binn = tmp_path / ".local" / "bin"
    binn.mkdir(parents=True)
    monkeypatch.setenv(guard.SHARED_BIN_DIRS_ENV, str(binn))
    kb.init_db()

    workspace = home / "kanban" / "workspaces" / "t_dead"
    workspace.mkdir(parents=True)
    shim = binn / "hermes"
    shim.write_text(
        f'#!/bin/sh\nexec {workspace}/.hermes/bin/hermes "$@"\n', encoding="utf-8"
    )

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="leaky", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind='scratch', workspace_path=? WHERE id=?",
                (str(workspace), tid),
            )
        kbw._cleanup_workspace(conn, tid)

    assert not workspace.exists()
    assert not shim.exists(), "teardown left a shared shim pointing into the workspace"
