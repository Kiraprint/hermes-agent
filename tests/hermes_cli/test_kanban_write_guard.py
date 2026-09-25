"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import importlib
import sys

import pytest

from hermes_cli import kanban_db


def _live_kdbc():
    """The ``kanban_db_connect`` module the guard actually patches.

    A module-level ``from hermes_cli import kanban_db_connect`` binds ONE
    module object, but the isolation fixtures reload that module during a
    session, leaving the local binding pointing at a stale object that
    ``sys.modules`` no longer holds. The guard patches whatever
    ``sys.modules[...]`` returns, so calling the stale binding's ``connect``
    silently bypasses the guard entirely — which is why these assertions
    passed in isolation and failed inside a full run. Resolve dynamically.
    """
    return sys.modules.get("hermes_cli.kanban_db_connect") or importlib.import_module(
        "hermes_cli.kanban_db_connect"
    )


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = _live_kdbc().connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    import tests.conftest as _conftest

    # Same stale-binding trap as _live_kdbc(): the guard's closure holds the
    # module object from sys.modules, so the resolver it calls has to be
    # patched on THAT object, not on whatever this file imported at collection.
    kdb = sys.modules.get("hermes_cli.kanban_db") or importlib.import_module(
        "hermes_cli.kanban_db"
    )
    real = _conftest._REAL_KANBAN_ROOTS[0]
    monkeypatch.setattr(kdb, "kanban_home", lambda: real)
    monkeypatch.setattr(kdb, "kanban_db_path", lambda board=None: real / "kanban.db")
    with pytest.raises(RuntimeError, match="kanban write guard"):
        _live_kdbc().connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    import tests.conftest as _conftest

    kdbc = _live_kdbc()
    for root in _conftest._REAL_KANBAN_ROOTS:
        with pytest.raises(RuntimeError, match="kanban write guard"):
            kdbc.connect(root / "kanban.db")


def test_connect_raises_when_deny_list_is_empty(monkeypatch):
    """No resolvable deny-root at all -> refuse, do not wave the write through.

    The guard reads ``_ktg._KANBAN_GUARD_DENY_ROOTS`` (published by the autouse
    conftest fixture) and falls back to the real platform ``~/.hermes``, so both
    have to be neutralised to reach this branch. Fail-closed is the whole point:
    an empty list used to be indistinguishable from "nothing to protect".
    """
    from hermes_cli import kanban_test_guard as _ktg

    monkeypatch.setattr(_ktg, "_KANBAN_GUARD_DENY_ROOTS", (), raising=False)
    monkeypatch.setattr(_ktg, "_real_platform_kanban_root", lambda: None, raising=False)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
        _live_kdbc().connect()


def test_pinned_kanban_db_env_is_in_deny_list(monkeypatch, tmp_path):
    """HERMES_KANBAN_DB — the var the dispatcher injects into every worker env.

    Regression for the fail-open: the old resolver only read
    HERMES_KANBAN_HOME, so with that unset a board pinned outside every hermes
    root resolved to a deny-list root that matched nothing and the write was
    allowed straight into the live board.
    """
    import tests.conftest as _conftest

    pinned = tmp_path / "boards" / "live" / "kanban.db"
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_KANBAN_DB", str(pinned))
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_KANBAN_OVERRIDE", "")
    roots = _conftest._capture_real_kanban_roots()

    assert pinned in roots
    # The guard consults the module-level deny-list, so point it at the roots
    # we just recomputed from the patched pre-sandbox env.
    monkeypatch.setattr(_conftest, "_REAL_KANBAN_ROOTS", roots)
    assert _conftest._is_real_kanban_target(pinned.resolve())


def test_pinned_kanban_board_slug_is_in_deny_list(monkeypatch, tmp_path):
    """HERMES_KANBAN_BOARD's board is denied even before its dir exists.

    A test that CREATES ``<real home>/kanban/boards/<slug>/kanban.db`` is the
    pollution the guard exists to stop, so the slug must be covered from the
    env value alone — not only by enumerating boards already on disk.
    """
    import tests.conftest as _conftest

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_KANBAN_BOARD", "not-created-yet")
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_HERMES_HOME", str(home))
    monkeypatch.setattr(
        _conftest, "_hermes_home_points_at_production", lambda _value: False
    )
    monkeypatch.setattr(
        _conftest, "_is_real_kanban_target", _conftest._is_real_kanban_target
    )
    roots = _conftest._capture_real_kanban_roots()

    target = (home / "kanban" / "boards" / "not-created-yet" / "kanban.db").resolve()
    assert not (home / "kanban" / "boards" / "not-created-yet").exists()
    assert any(target == r or target.is_relative_to(r) for r in roots), (
        "HERMES_KANBAN_BOARD slug missing from deny-list: " f"{roots}"
    )


def test_hermes_home_root_is_not_broadened_to_its_parent(monkeypatch, tmp_path):
    """The hermes home contributes ITSELF, never its parent.

    Adding the parent of the home root would deny ``/opt/data`` — a whole
    filesystem subtree — and block unrelated legitimate writes.
    """
    import tests.conftest as _conftest

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_KANBAN_DB", "")
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_KANBAN_OVERRIDE", "")
    monkeypatch.setattr(_conftest, "_PRE_SANDBOX_HERMES_HOME", str(home))
    monkeypatch.setattr(
        _conftest, "_hermes_home_points_at_production", lambda _value: False
    )
    roots = _conftest._capture_real_kanban_roots()

    assert home.resolve() in roots
    assert tmp_path.resolve() not in roots

