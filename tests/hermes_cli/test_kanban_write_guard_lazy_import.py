"""The kanban write guard must arm even when nothing imported kanban yet.

This file deliberately imports NO kanban module at module level. That is the
whole point: on the pre-fix conftest the autouse ``_kanban_write_guard``
probed ``sys.modules`` and returned early when the module was absent, so a
test that reached for kanban lazily ran completely unguarded and could
write to the real board.

Run this file ALONE to reproduce the original defect:

    pytest tests/hermes_cli/test_kanban_write_guard_lazy_import.py

On the pre-fix tree the first test writes a row into the real board; with
the fix it is refused. Importing here (inside the tests) is deliberate --
module-level imports would pre-empt the very condition being tested.
"""

from __future__ import annotations

import pytest


def test_lazy_first_import_still_meets_a_guarded_connect():
    """A body-level import must find ``connect`` already wrapped."""
    from hermes_cli import kanban_db_connect as kbc

    assert "guarded" in kbc.connect.__qualname__, (
        "kanban write guard did not arm: nothing imported kanban_db_connect "
        "before the guard ran, so the guard no-op'd and this test would be "
        "writing to the real board unguarded"
    )


def test_lazy_import_cannot_reach_a_real_board():
    """The lazy path refuses a write under a real kanban root."""
    import tests.conftest as _conftest
    from hermes_cli import kanban_db_connect as kbc

    target = _conftest._REAL_KANBAN_ROOT / "kanban.db"

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(target)


def test_guard_arms_before_any_test_body_runs():
    """The arming decision belongs to the fixture, not to the test.

    This test imports nothing kanban-related on the unguarded path either;
    it only asserts the invariant the fixture is responsible for.
    """
    import tests.conftest as _conftest

    assert _conftest._REAL_KANBAN_ROOT, (
        "the guard has no root to deny; an unresolvable root must fail closed"
    )
