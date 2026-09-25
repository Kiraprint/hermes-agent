"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc


# ── Regression: a board pinned OUTSIDE ~/.hermes must still be denied ──────
# The 44 phantom cards on the live board came from a dispatcher worker whose
# board was pinned with HERMES_KANBAN_DB to a path outside every root the
# single-root resolver could see, so `relative_to` raised ValueError and the
# guard read that as "not the real board" (fail-open).
#
# These tests drive the capture helpers with a controlled pre-sandbox env
# snapshot, because the module-level globals are frozen at conftest import.


def _recapture(monkeypatch, *, kanban_db_path=None, board=None, kanban_home=None,
               hermes_home=None):
    """Recompute the deny-roots from an explicit pre-sandbox env snapshot."""
    import tests.conftest as c

    monkeypatch.setattr(c, "_PRE_SANDBOX_KANBAN_DB", kanban_db_path or "", raising=False)
    monkeypatch.setattr(c, "_PRE_SANDBOX_KANBAN_BOARD", board or "", raising=False)
    monkeypatch.setattr(c, "_PRE_SANDBOX_KANBAN_OVERRIDE", kanban_home or "", raising=False)
    monkeypatch.setattr(c, "_PRE_SANDBOX_HERMES_HOME", hermes_home or "", raising=False)
    return c._capture_real_kanban_deny_roots()


def test_deny_roots_cover_board_pinned_outside_home(monkeypatch, tmp_path):
    """A HERMES_KANBAN_DB pin outside ~/.hermes lands in the deny-list."""
    outside = tmp_path / "elsewhere" / "kanban" / "boards" / "live"
    outside.mkdir(parents=True)
    pinned_db = outside / "kanban.db"

    roots = _recapture(monkeypatch, kanban_db_path=str(pinned_db), board="live")

    assert pinned_db.resolve() in [r for r in roots] or any(
        _is_under(pinned_db, r) for r in roots
    ), f"pinned live board {pinned_db} not covered by deny-roots {roots}"


def test_deny_roots_cover_whole_boards_tree(monkeypatch, tmp_path):
    """The boards tree is denied, so sibling boards are covered too."""
    boards = tmp_path / "data" / "kanban" / "boards"
    board_dir = boards / "live"
    board_dir.mkdir(parents=True)

    roots = _recapture(monkeypatch, kanban_db_path=str(board_dir / "kanban.db"))

    assert any(_is_under(board_dir, r) for r in roots), roots
    # A sibling board of the same live install is denied as well.
    assert any(_is_under(boards / "other", r) for r in roots), roots


def test_hermetic_sibling_tempdir_stays_allowed(monkeypatch, tmp_path):
    """The #69385 regression: hermetic tests must remain a no-op.

    A test that moves HERMES_HOME to its own tempdir resolves to a path under no
    deny entry, so the guard must not reject it.
    """
    hermetic = tmp_path / "pytest-999" / "test_x0" / "hermes_test" / "kanban.db"

    roots = _recapture(
        monkeypatch,
        kanban_db_path="/opt/data/kanban/boards/factory/kanban.db",
    )

    assert not any(_is_under(hermetic, r) for r in roots), (
        f"hermetic tempdir {hermetic} must not be denied by {roots}"
    )


def test_deny_roots_never_include_filesystem_root(monkeypatch):
    """The filesystem root is dropped — denying it would block every test."""
    roots = _recapture(monkeypatch, kanban_db_path="/kanban.db")
    from pathlib import Path

    assert Path("/") not in roots, roots


def test_guard_fails_closed_on_empty_deny_list(monkeypatch):
    """An unresolvable snapshot must refuse writes, not permit them.

    With the deny-list empty the guard cannot tell a live board from a tempdir,
    so it must raise rather than fall through to the real connect().
    """
    import tests.conftest as c
    from hermes_cli import kanban_db_connect as _kdbc

    assert _kdbc.connect is not None
    monkeypatch.setattr(c, "_REAL_KANBAN_DENY_ROOTS", (), raising=False)

    # Invoke the fixture's underlying function directly (pytest wraps it in a
    # FixtureFunctionDefinition). ``_hermetic_environment`` is an autouse
    # fixture already applied to this test, so None is correct here — the guard
    # does not use it.
    func = c._kanban_write_guard.__wrapped__

    with pytest.raises(RuntimeError, match="no REAL kanban deny-root"):
        func(None, monkeypatch)


def _is_under(path, root) -> bool:
    from pathlib import Path

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


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
    import tests.conftest as _conftest

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: _conftest._REAL_KANBAN_ROOT
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: _conftest._REAL_KANBAN_ROOT / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    import tests.conftest as _conftest

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(_conftest._REAL_KANBAN_ROOT / "kanban.db")
