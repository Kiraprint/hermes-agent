"""Fail-closed barrier against launcher shims leaking out of worker workspaces.

A kanban worker runs Hermes from an EPHEMERAL checkout: its scratch workspace
(``<kanban>/boards/<slug>/workspaces/<task>``, or a ``hermes-wt`` worktree
inside it). Boot maintenance publishes the PATH conveniences -- ``hermes``,
``hermes-acp``, ``hermes-agent`` -- through
:func:`hermes_cli._launchers.expose_cli`, and when that ephemeral checkout is
the install root those shims hard-code a path *inside the workspace*:

    $HOME/.local/bin/hermes -> /opt/data/kanban/boards/factory/workspaces/t_7180bf50/.hermes/bin/hermes

The workspace is deleted when the task completes, so the shared entry dangles
and ``hermes`` stops resolving for every other shell that has that bin dir on
PATH (t_7180bf50 left three such shims; t_a1a9f616 left a fourth in the
per-profile bin dir). The leak is invisible to the worker -- it only breaks
*other* sessions later.

This module is the barrier, in two halves:

* :func:`is_ephemeral_root` + :func:`shared_bin_dirs` -- the fail-closed
  predicate :func:`hermes_cli._launchers.expose_cli` consults *before* it
  touches a shared bin dir. An ephemeral checkout keeps its own
  ``<root>/.hermes/bin`` launchers and is refused ownership of a shared
  PATH entry.
* :func:`scrub_shared_bin_leaks` -- removes entries that already point into
  kanban-managed workspaces (dangling or not). It runs at worker teardown
  (:func:`hermes_cli.kanban_db_workspace._cleanup_workspace`) and on every
  launcher publication, so leaks are cleaned even by builds that predate
  this guard. It is also available standalone::

      python -m hermes_cli.launcher_leak_guard --scrub --dry-run
      python -m hermes_cli.launcher_leak_guard --scrub --repair-hermes /opt/hermes/.venv/bin/hermes

Policy, deliberately fail-closed: nothing in a shared bin dir may reference a
kanban workspaces root. Only entries that provably do (a symlink whose target
lands under such a root, or a text launcher whose payload mentions one) are
touched; everything else in those directories is left alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

log = logging.getLogger(__name__)

#: Commands an install publishes for the user (:data:`hermes_cli._launchers.ENTRY_POINTS`).
LAUNCHER_NAMES = ("hermes", "hermes-agent", "hermes-acp")

#: Text launchers are tiny; a larger file that mentions a workspace path is
#: reported as suspicious rather than deleted.
_MAX_SCAN_BYTES = 1 << 20

#: A workspace path, whatever the board layout is: ``<root>/kanban/workspaces``
#: for the default board, ``<root>/kanban/boards/<slug>/workspaces`` otherwise.
_WORKSPACE_TEXT_RE = re.compile(r"/kanban/(?:boards/[^/\s\"']+/)?workspaces/")
_KANBAN_PART = "kanban"
_WORKSPACES_PART = "workspaces"

#: Optional operator override: ``os.pathsep``-separated list of bin dirs the
#: sweep and the scrub CLI are allowed to touch. Unset (the normal case) means
#: the shared defaults below; set, it *replaces* them so a deployment (or a
#: test) can pin the blast radius instead of trusting a guess at the layout.
SHARED_BIN_DIRS_ENV = "HERMES_LAUNCHER_LEAK_GUARD_DIRS"


def _norm(path: Path | str) -> Path:
    """Lexical normalization: never resolve symlinks, never touch the disk.

    Workspace paths are routinely *deleted* by the time we judge them, so
    ``resolve()`` (which collapses dangling links to their raw text anyway) or
    ``os.path.realpath`` must not be the only view we have.
    """
    return Path(os.path.normpath(str(path)))


def _path_parts(path: Path | str) -> tuple[str, ...]:
    parts = _norm(path).parts
    return parts[1:] if parts and parts[0] == os.sep else parts


def path_looks_like_kanban_workspace(path: Path | str) -> bool:
    """Lexical verdict for *path*: does it live under a kanban ``workspaces`` root?

    Layout-agnostic on purpose -- it must answer for a path that no longer
    exists (dangling shim targets) and for boards whose names we do not know.
    ``/srv/workspaces/proj`` (no ``kanban`` component) stays clean.
    """
    parts = _path_parts(path)
    if _KANBAN_PART not in parts or _WORKSPACES_PART not in parts:
        return False
    return parts.index(_WORKSPACES_PART) > parts.index(_KANBAN_PART)


def _env_workspace_roots() -> list[Path]:
    """Workspaces roots named by the environment, for non-standard layouts.

    ``HERMES_KANBAN_WORKSPACES_ROOT`` is the authoritative override (the
    dispatcher injects it into every worker); the DB/home vars give the board
    layout when only those are visible. Missing roots are kept: the lexical
    helpers work on text.
    """
    roots: list[Path] = []
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        roots.append(Path(override).expanduser())
    db = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if db:
        parent = Path(db).expanduser().parent
        # ``<board>/kanban.db`` for named boards, ``<kanban_home>/kanban.db``
        # for the default board -- both keep their workspaces dir alongside.
        roots.extend([parent / _WORKSPACES_PART, parent.parent / _WORKSPACES_PART])
    home = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if home:
        roots.append(Path(home).expanduser() / "kanban" / _WORKSPACES_PART)
    return roots


def workspace_roots(hint: Path | str | None = None, extra: Iterable[Path | str] = ()) -> list[Path]:
    """Every kanban workspaces root we can name, deduped, order preserved."""
    candidates: list[Path] = []
    if hint is not None:
        candidates.append(Path(hint).expanduser())
    candidates.extend(Path(item).expanduser() for item in extra)
    candidates.extend(_env_workspace_roots())
    seen: set[str] = set()
    roots: list[Path] = []
    for candidate in candidates:
        key = str(_norm(candidate))
        if key and key not in seen:
            seen.add(key)
            roots.append(_norm(candidate))
    return roots


def is_inside_workspace(path: Path | str, roots: Sequence[Path | str] | None = None) -> bool:
    """Is *path* an ephemeral kanban workspace (or a descendant of one)?"""
    normalized = _norm(path)
    if path_looks_like_kanban_workspace(normalized):
        return True
    for root in roots if roots is not None else workspace_roots():
        root = _norm(root)
        if normalized == root or normalized.is_relative_to(root):
            return True
    return False


def is_ephemeral_root(root: Path | str, roots: Sequence[Path | str] | None = None) -> bool:
    """Would publishing *root* as a shared install root leak an ephemeral path?

    This is the gate :func:`expose_cli` applies before it writes anything into
    a shared bin directory. It is deliberately generous: a false positive only
    costs a worker its own ``~/.local/bin`` convenience (it keeps
    ``<root>/.hermes/bin`` and calls Hermes by absolute path), while a false
    negative leaves a dangling ``hermes`` behind for every later session.
    """
    return is_inside_workspace(root, roots)


def shared_bin_dirs(extra: Iterable[Path | str] = ()) -> list[Path]:
    """The bin directories every install shares (home bin first).

    ``$HOME/.local/bin`` is the one a worker-owned shim actually lands in; the
    per-machine install bin and ``/usr/local/bin`` are included so the scrub
    sees the FHS layout too. Roots are *not* resolved: a caller inside a
    workspace must still see the real global bin dir.
    """
    override = os.environ.get(SHARED_BIN_DIRS_ENV, "").strip()
    if override:
        dirs: list[Path] = [
            Path(part).expanduser() for part in override.split(os.pathsep) if part.strip()
        ]
    else:
        dirs = [Path.home() / ".local" / "bin"]
        try:
            from hermes_constants import get_default_hermes_root

            dirs.append(get_default_hermes_root() / "bin")
        except Exception as exc:  # noqa: BLE001 - a broken home must not stop the scrub
            log.debug("shared bin: default hermes root unavailable: %s", exc)
        dirs.append(Path("/usr/local/bin"))
    dirs.extend(Path(d).expanduser() for d in extra)
    seen: set[str] = set()
    unique: list[Path] = []
    for directory in dirs:
        key = str(_norm(directory))
        if key not in seen:
            seen.add(key)
            unique.append(Path(directory))
    return unique


def _entry_text(entry: Path) -> Optional[str]:
    """Decoded payload of a small text launcher, or ``None`` when undecidable."""
    try:
        stat = entry.stat()
    except OSError:
        return None
    if stat.st_size > _MAX_SCAN_BYTES:
        return None
    try:
        raw = entry.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - defensive; decode above never raises
        return None


def leak_reason(entry: Path, roots: Sequence[Path | str] | None = None) -> Optional[str]:
    """Why *entry* must not stay in a shared bin dir, else ``None``.

    Returns ``"symlink:<target>"`` for a link into a workspace (dangling links
    included -- that is the failure this guard exists for),
    ``"content"`` for a text launcher whose payload references one, and
    ``"suspicious-binary"`` when a binary-like file mentions one (reported,
    never deleted).
    """
    try:
        if entry.is_symlink():
            target = os.readlink(entry)
            resolved = Path(target)
            if not resolved.is_absolute():
                resolved = entry.parent / resolved
            return f"symlink:{target}" if is_inside_workspace(resolved, roots) else None
    except OSError as exc:
        log.debug("shared bin: cannot inspect %s: %s", entry, exc)
        return None

    try:
        if entry.is_dir():
            return None
        raw = entry.read_bytes() if entry.stat().st_size <= _MAX_SCAN_BYTES else b""
    except OSError:
        return None
    if not raw:
        return None

    if b"\0" in raw:
        try:
            blob = raw.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
        return "suspicious-binary" if _WORKSPACE_TEXT_RE.search(blob) else None

    text = raw.decode("utf-8", "replace")
    if _WORKSPACE_TEXT_RE.search(text):
        return "content"
    for root in workspace_roots() if roots is None else roots:
        if str(_norm(root)) in text:
            return "content"
    return None


def scrub_shared_bin_leaks(
    *,
    roots: Sequence[Path | str] | None = None,
    dirs: Sequence[Path | str] | None = None,
    dry_run: bool = False,
    repair_hermes: Path | str | None = None,
) -> dict:
    """Remove workspace-pointing entries from the shared bin directories.

    Best-effort by contract: every failure is collected, never raised -- this
    runs from worker teardown, where a cleanup problem must not fail the task.
    ``repair_hermes`` replaces a removed ``hermes`` command with a symlink to a
    durable CLI (typically ``/opt/hermes/.venv/bin/hermes``), so scrubbing does
    not leave a shell without ``hermes`` at all.
    """
    known_roots = workspace_roots(extra=roots or ())
    report: dict = {
        "roots": [str(root) for root in known_roots],
        "dirs": [],
        "removed": [],
        "repaired": [],
        "suspicious": [],
        "kept": 0,
        "dry_run": dry_run,
        "errors": [],
    }
    targets = [Path(d) for d in dirs] if dirs is not None else shared_bin_dirs()
    for directory in targets:
        try:
            if not directory.is_dir():
                continue
            report["dirs"].append(str(directory))
            entries = sorted(directory.iterdir())
        except OSError as exc:
            report["errors"].append(f"{directory}: {exc}")
            continue
        for entry in entries:
            try:
                reason = leak_reason(entry, known_roots)
                if reason is None:
                    report["kept"] += 1
                    continue
                if reason == "suspicious-binary":
                    report["suspicious"].append(f"{entry} ({reason})")
                    continue
                if dry_run:
                    report["removed"].append(f"{entry} ({reason})")
                    continue
                entry.unlink()
                report["removed"].append(f"{entry} ({reason})")
                log.warning("removed leaked launcher %s (%s)", entry, reason)
                if repair_hermes is not None and entry.name == "hermes":
                    link = Path(repair_hermes).expanduser()
                    os.symlink(link, entry)
                    report["repaired"].append(f"{entry} -> {link}")
            except OSError as exc:
                report["errors"].append(f"{entry}: {exc}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Operator entry point: ``python -m hermes_cli.launcher_leak_guard --scrub``."""
    parser = argparse.ArgumentParser(prog="hermes_cli.launcher_leak_guard")
    parser.add_argument("--scrub", action="store_true",
                        help="remove workspace-pointing launchers from shared bin dirs")
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--bin-dir", action="append", default=[],
                        help="shared bin dir to inspect (repeatable; default: all)")
    parser.add_argument("--workspace-root", action="append", default=[],
                        help="extra kanban workspaces root to treat as ephemeral")
    parser.add_argument("--repair-hermes", default=None,
                        help="symlink a scrubbed `hermes` to this durable CLI")
    args = parser.parse_args(argv)

    if not args.scrub:
        parser.print_help()
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = scrub_shared_bin_leaks(
        roots=args.workspace_root or None,
        dirs=args.bin_dir or None,
        dry_run=args.dry_run,
        repair_hermes=args.repair_hermes,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if report["errors"] else 0
    verb = "would remove" if args.dry_run else "removed"
    for line in report["removed"]:
        print(f"{verb}: {line}")
    for line in report["repaired"]:
        print(f"repaired: {line}")
    for line in report["suspicious"]:
        print(f"suspicious (kept): {line}")
    for line in report["errors"]:
        print(f"error: {line}", file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
