"""Live-kanban-board test-isolation guard.

Why this lives in production code and not in ``tests/conftest.py``
------------------------------------------------------------------
The guard used to be a ``monkeypatch`` of
``hermes_cli.kanban_db_connect.connect`` installed by an autouse fixture. That
placement has a hole that no amount of deny-list tuning can close: the fixture
only patches when ``hermes_cli.kanban_db_connect`` is **already in
``sys.modules``**, so a test that imports the kanban stack *inside* the test
body (a lazy import) gets the **unpatched** ``connect`` and writes to the live
board with no protection at all. Reproduced against #69283's follow-up PRs:
the deny-list was correct and still the write sailed through.

So enforcement moves to the call site, mirroring
``hermes_state._ensure_test_isolation`` (#82770): a pytest-context process
(env OR ancestry) can never open a board that the deny-list says is real. The
check reads its knobs at CALL time, which makes import order irrelevant.

The deny-roots themselves still come from the conftest, because only the
conftest can see the **pre-sandbox** environment: the hermetic fixture deletes
``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / ``HERMES_KANBAN_HOME`` before
any test body runs, so by call time the real pin is already gone from the env.
``tests/conftest.py`` therefore hands the resolved roots over through
:data:`_KANBAN_GUARD_DENY_ROOTS`; this module adds a self-sufficient
``~/.hermes`` fallback for processes where the conftest never ran.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]


#: Env twin of :data:`_KANBAN_GUARD_BYPASS` for child processes. A module global
#: cannot cross a process boundary, so a pytest-spawned child reads this instead.
_KANBAN_GUARD_BYPASS_ENV = "HERMES_KANBAN_DB_GUARD_BYPASS"

#: Escape hatch, flipped by ``@pytest.mark.live_system_guard_bypass``.
_KANBAN_GUARD_BYPASS = False

#: Pre-sandbox real roots, injected by the hermetic conftest. See module docstring.
_KANBAN_GUARD_DENY_ROOTS: Tuple[Path, ...] = ()

_PYTEST_ANCESTOR: Optional[bool] = None


def _has_pytest_ancestor() -> bool:
    """True when a pytest process is an ancestor of this one (memoised).

    Ancestry is the one test-context signal that survives an env rebuild in a
    child process, which is precisely the state in which a naive write to the
    live board happens (#82770).
    """
    global _PYTEST_ANCESTOR
    if _PYTEST_ANCESTOR is not None:
        return _PYTEST_ANCESTOR
    found = False
    if psutil is not None:
        try:
            import psutil as _psutil

            me = _psutil.Process()
            for parent in me.parents():
                cmdline = " ".join(parent.cmdline() or ())
                if "pytest" in cmdline or "py.test" in cmdline:
                    found = True
                    break
        except Exception:
            found = False
    _PYTEST_ANCESTOR = found
    return found


def _running_under_pytest() -> bool:
    """True when this process is itself pytest (env signal, not ancestry)."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


def _in_test_context() -> bool:
    """Test run by environment or ancestry (env checked first)."""
    return _running_under_pytest() or _has_pytest_ancestor()


def _real_platform_kanban_root() -> Optional[Path]:
    """The REAL platform-default Hermes root, without trusting the sandbox.

    ``Path.home()`` is monkeypatched by the hermetic conftest and
    ``hermes_constants`` reads the already-redirected ``HERMES_HOME``, so both
    would describe the throwaway tempdir. ``expanduser("~")`` reads
    HOME/passwd, which the conftest deliberately does NOT rewrite (see the
    NOTE in ``_hermetic_environment``).
    """
    try:
        home = Path(os.path.expanduser("~"))
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "").strip()
            root = Path(base) / "hermes" if base else home / "AppData" / "Local" / "hermes"
        else:
            root = home / ".hermes"
        return root.resolve()
    except Exception:
        return None


def _resolve_roots() -> list[Path]:
    """Deny-roots to enforce: conftest-injected plus the self-sufficient fallback."""
    roots: list[Path] = []
    for value in _KANBAN_GUARD_DENY_ROOTS:
        try:
            roots.append(Path(value).expanduser().resolve())
        except Exception:
            continue
    fallback = _real_platform_kanban_root()
    if fallback is not None:
        roots.append(fallback)
    # Deduplicate, keep order stable for readable failure messages.
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _is_real_kanban_target(resolved: Path, root: Path) -> bool:
    """True when *resolved* is *root* itself or lives under it."""
    if resolved == root:
        return True
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_kanban_test_isolation(db_path: Path) -> None:
    """Raise before opening a connection when a test-context process targets a real board.

    Called at the top of ``kanban_db_connect.connect`` — the single choke point
    every board open goes through. Raising HERE (rather than from a fixture)
    is what makes the guard immune to import order: a lazy import resolves the
    same ``connect`` function this check already lives in.
    """
    if (
        _KANBAN_GUARD_BYPASS
        or os.environ.get(_KANBAN_GUARD_BYPASS_ENV)
        or not _in_test_context()
    ):
        return
    try:
        resolved = Path(db_path).expanduser().resolve()
    except Exception:
        return

    roots = _resolve_roots()
    if not roots:
        # Fail CLOSED. An empty deny-list used to be indistinguishable from
        # "nothing to protect", and the guard then passed every write through
        # silently. ``_real_platform_kanban_root`` should make this
        # unreachable, but if it ever is, refuse rather than pollute.
        raise RuntimeError(
            "kanban write guard: FAIL-CLOSED — a pytest-context process asked to "
            f"open {resolved}, but no real-board deny-root could be resolved. "
            f"Refusing rather than risk writing a live board. Set "
            f"{_KANBAN_GUARD_BYPASS_ENV}=1 to override."
        )

    for root in roots:
        if _is_real_kanban_target(resolved, root):
            raise RuntimeError(
                "kanban write guard: refusing to open the REAL kanban board "
                f"({resolved}) from a test-context process (deny-root: {root}). "
                "Tests must use a temporary HERMES_HOME. If this is a "
                "legitimate live-system test, mark it "
                "@pytest.mark.live_system_guard_bypass."
            )
