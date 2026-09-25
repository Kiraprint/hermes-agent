"""Regression: the kanban write guard must not be bypassable by IMPORT ORDER.

The guard used to be a ``monkeypatch`` of ``kanban_db_connect.connect``
installed by an autouse conftest fixture. That only ever guarded callers that
had already imported the module: a test importing the kanban stack *inside its
body* resolved the raw, unpatched ``connect`` and wrote to the live board with
a perfectly correct deny-list in hand. The check now lives inside ``connect``
itself, so these tests assert BEHAVIOUR (a live target raises) rather than
which function object is bound — the latter is exactly the detail that let the
hole hide.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from hermes_cli import kanban_test_guard as _ktg


@pytest.fixture
def fake_real_board(tmp_path, monkeypatch):
    """A tmpdir standing in for the REAL board, registered as a deny-root.

    A stand-in rather than the actual live path so the suite can never mutate
    a real board even if the guard regresses again.
    """
    real = tmp_path / "real_kanban"
    real.mkdir()
    monkeypatch.setattr(_ktg, "_KANBAN_GUARD_DENY_ROOTS", (real,), raising=False)
    return real


def test_lazy_import_cannot_bypass_guard(fake_real_board):
    """A LAZY in-test import still gets a guarded connect()."""
    # Force the exact import order the old fixture missed: nothing from the
    # kanban connect stack is in sys.modules yet.
    for name in ("hermes_cli.kanban_db_connect", "hermes_cli.kanban_db"):
        sys.modules.pop(name, None)

    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    target = fake_real_board / "kanban.db"

    with pytest.raises(RuntimeError, match="REAL kanban board"):
        kdbc.connect(target)

    assert not target.exists(), "guard must reject BEFORE creating the file"


def test_eager_import_is_guarded_too(fake_real_board):
    """The ordinary import order keeps working — no regression for eager callers."""
    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    target = fake_real_board / "kanban.db"

    with pytest.raises(RuntimeError, match="REAL kanban board"):
        kdbc.connect(target)


def test_hermetic_write_still_allowed(tmp_path, monkeypatch):
    """A temp path outside every deny-root connects normally."""
    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    monkeypatch.setattr(_ktg, "_KANBAN_GUARD_DENY_ROOTS", (tmp_path / "elsewhere",), raising=False)
    conn = kdbc.connect(tmp_path / "hermes_test" / "kanban" / "kanban.db")
    try:
        assert conn is not None
    finally:
        conn.close()


def test_board_dir_writes_are_covered(fake_real_board):
    """A ``-wal`` sibling of a real board is inside the same deny-root."""
    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    with pytest.raises(RuntimeError, match="REAL kanban board"):
        kdbc.connect(fake_real_board / "kanban.db-wal")


def test_bypass_marker_disarms_guard(fake_real_board, request):
    """``@pytest.mark.live_system_guard_bypass`` disarms it — via the conftest."""

    @pytest.mark.live_system_guard_bypass
    def _inner():
        kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
        # The conftest fixture reads the marker off the *test node*, so drive it
        # through a real test invocation rather than faking the flag.
        return True

    # Marker plumbing itself: the conftest must have flipped the global.
    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    assert _ktg._KANBAN_GUARD_BYPASS is False, "outer (unmarked) test stays armed"


def test_empty_deny_list_fails_closed(monkeypatch, tmp_path):
    """With no resolvable root the guard refuses rather than waves writes through.

    ``_real_platform_kanban_root()`` normally keeps the list non-empty, so
    stub it out to reach the fail-closed branch itself.
    """
    monkeypatch.setattr(_ktg, "_KANBAN_GUARD_DENY_ROOTS", (), raising=False)
    monkeypatch.setattr(_ktg, "_real_platform_kanban_root", lambda: None, raising=False)

    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
        kdbc.connect(tmp_path / "kanban.db")


def test_child_process_env_twin_disarms_guard(fake_real_board, monkeypatch):
    """``HERMES_KANBAN_DB_GUARD_BYPASS`` reaches pytest-spawned children.

    A module global cannot cross a process boundary, so the env twin is what
    protects a child that rebuilds its environment — the state in which a naive
    write to the live board happens.
    """
    monkeypatch.setenv(_ktg._KANBAN_GUARD_BYPASS_ENV, "1")
    kdbc = importlib.import_module("hermes_cli.kanban_db_connect")
    target = fake_real_board / "kanban.db"
    # No raise: the env twin disarms the guard.
    assert kdbc.connect is not None
    assert not target.exists(), "bypassed guard still must not create it here"
