"""#69283: the kanban write guard refuses writes to the live board.

Covered here:
* live board paths are refused — explicit ``db_path``, resolved
  ``kanban_db_path()`` and the ``board=`` kwarg alike;
* hermetic paths that merely live BESIDE the live root keep working: the
  deny-list must not swallow sibling tempdirs (over-block regression);
* with no pinnable live root the guard fails closed instead of allowing
  everything it could not classify.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc

# These probe the kanban guard against the real root on purpose.
pytestmark = pytest.mark.allow_real_home_io


def _conftest_module():
    import tests.conftest as conftest

    return conftest


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kbc.connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    conftest = _conftest_module()

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: conftest._REAL_KANBAN_ROOT
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: conftest._REAL_KANBAN_ROOT / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    conftest = _conftest_module()

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(conftest._REAL_KANBAN_ROOT / "kanban.db")


def test_connect_raises_for_board_kwarg_resolving_into_live_tree(monkeypatch):
    """board= reaches the resolver and is refused like any other live path."""
    conftest = _conftest_module()

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: conftest._REAL_KANBAN_ROOT
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(board="factory")


def test_every_live_deny_entry_is_refused(monkeypatch):
    """Each captured live entry blocks the board DBs it owns — and only them."""
    conftest = _conftest_module()
    # This is a deny-list unit test: pin the fail-closed rule ON so a scrubbed
    # runner env (no HERMES_* pin) cannot mask what is being asserted here.
    monkeypatch.setattr(conftest, "_KANBAN_ROOT_DETERMINED", True)

    assert conftest._REAL_KANBAN_DENY_ENTRIES, "live deny-list must not be empty"
    for kind, entry in conftest._REAL_KANBAN_DENY_ENTRIES:
        if kind == "exact":
            with pytest.raises(RuntimeError, match="kanban_write_guard"):
                conftest._kanban_write_guard_check(entry)
            continue
        with pytest.raises(RuntimeError, match="kanban_write_guard"):
            conftest._kanban_write_guard_check(entry / "factory" / "kanban.db")
        # Scratch workspaces of the very same board dir stay writable: task
        # worktrees run from there (over-block regression).
        conftest._kanban_write_guard_check(
            entry / "factory" / "workspaces" / "t_x" / "hermes-wt" / "kanban.db"
        )
        # So do DB files that merely sit one level deeper than a board DB.
        conftest._kanban_write_guard_check(entry / "factory" / "logs" / "copy.db")


def test_hermetic_siblings_of_the_live_root_still_writable(monkeypatch):
    """A whole-directory prefix would block these — the deny-list must not."""
    conftest = _conftest_module()
    # Deny-list scope only: the fail-closed rule would reject synthetic paths
    # in a scrubbed runner env, which is not what this test is about.
    monkeypatch.setattr(conftest, "_KANBAN_ROOT_DETERMINED", True)
    root = conftest._REAL_KANBAN_ROOT

    for hermetic in (
        root / "profiles" / "factory-dev" / "cache" / "scratch" / "x" / "kanban.db",
        root / "tmp" / "sandbox" / "kanban.db",
        Path(tempfile.gettempdir()) / "hermetic" / "kanban.db",
    ):
        # Must not raise: sibling dirs, not the live board tree.
        conftest._kanban_write_guard_check(hermetic)


def test_fail_closed_when_the_live_root_cannot_be_pinned(monkeypatch):
    """Undetermined live root: only provably hermetic paths stay writable."""
    conftest = _conftest_module()

    monkeypatch.setattr(conftest, "_KANBAN_ROOT_DETERMINED", False)
    outside = Path.cwd() / "some_sandbox_dir" / "kanban.db"
    with pytest.raises(RuntimeError, match="fail-closed"):
        conftest._kanban_write_guard_check(outside)
    # Hermetic paths stay writable under fail-closed, too.
    conftest._kanban_write_guard_check(
        Path(tempfile.gettempdir()) / "hermetic" / "kanban.db"
    )
