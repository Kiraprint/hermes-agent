"""#69283: the write guard is installed for EVERY session, unconditionally.

Historically the guard only patched ``connect`` when the kanban modules were
already sitting in ``sys.modules`` (a probe, not an import), so a run that
started with a clean module table ran unguarded and wrote real cards into the
live board — both guard tests then reported DID NOT RAISE.

This file deliberately does NOT import ``hermes_cli.kanban_db`` at module
level: it inspects the marker the guard leaves behind, proving the patch
lands at conftest import time — before pytest imports any test module, and
therefore also before every session and every test body.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_guard_was_installed_at_conftest_import_time():
    """Not at fixture setup: at import, before any test module binds names."""
    import tests.conftest as conftest

    assert conftest._KANBAN_GUARD_INSTALLED_AT_IMPORT is True


def test_connect_is_guarded_without_this_module_importing_kanban_first():
    import hermes_cli.kanban_db_connect as kbc

    assert getattr(kbc, "_hermes_test_write_guard", False) is True
    original = getattr(kbc, "_hermes_test_write_guard_orig", None)
    assert original is not None, "original connect must be kept for restore"
    assert kbc.connect is not original, "connect must be the guarded wrapper"
    assert kbc.connect.__name__ == "connect"


def test_module_level_value_import_binds_the_guarded_connect():
    """A test file doing ``from ... import connect`` must not bypass the guard."""
    from hermes_cli.kanban_db_connect import connect

    assert getattr(connect, "_hermes_test_write_guard", False) is True


def test_guarded_connect_refuses_the_pinned_live_db():
    import hermes_cli.kanban_db_connect as kbc

    import tests.conftest as conftest

    live = Path(
        conftest._PRE_SANDBOX_KANBAN_DB or (conftest._REAL_KANBAN_ROOT / "kanban.db")
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(live)
