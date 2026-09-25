"""Content-free fork-sync health projection (fork ↔ upstream divergence).

The nightly ``fork_sync_nightly.sh`` runner leaves two artefacts this module
reads — both read-only and best-effort:

* its log (``$FORK_SYNC_LOG``, else ``$HERMES_HOME/logs/fork-sync.log``), whose
  contract is one terminal ``RESULT ...`` line per run: ``clean sync`` (rc=0),
  ``created task <id>`` (rc=2), ``push failed`` (rc=3), ``error`` (rc=1).  The
  pre-hardening script wrote the same outcomes as free-form lines
  (``rebase-sync done`` / ``conflict task opened`` / ``ERROR: rebase ok but push
  failed``); both shapes are recognised so a host that has not deployed the
  hardened script yet still reports a real status instead of ``unknown``.
* the board, where an unresolved rebase conflict opens exactly one escalation
  task (``idempotency_key`` prefix ``fork-sync-conflict-``).  An open escalation
  task is what separates "diverged but owned by a human" (degraded) from
  "diverged and nobody is looking" (unhealthy).

Status vocabulary (bounded, exported as an attribute):

* ``healthy``   — last run synced and is fresh;
* ``degraded``  — synced but stale, or a conflict escalated to an open board task;
* ``unhealthy`` — last run failed, or a conflict has no open escalation task;
* ``unknown``   — no runner artefacts on this host, or a read broke.

Every read is independently fail-open: a missing log, an unreadable board or a
raising reader degrades the *reported* status and never propagates.  This feed
reports liveness for the gateway; it is not part of the dispatch path, so it can
never stall the dispatcher or the ready queue.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from agent.monitoring.gateway_health import GatewayMetric, _safe_metric_value

logger = logging.getLogger("gateway.fork_sync")

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"

KNOWN_STATUSES = frozenset({HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN})
# Bounded reason vocabulary — never a free-form string (it is exported as an attribute).
KNOWN_REASONS = frozenset({
    "clean_sync", "stale_sync", "conflict_escalated", "conflict_unattended",
    "push_failed", "run_failed", "no_state", "reader_error",
})
# ``clean`` / ``conflict`` / ``push_failed`` / ``error`` / ``unknown``.
KNOWN_RESULTS = frozenset({"clean", "conflict", "push_failed", "error", "unknown"})

DEFAULT_STALE_AFTER_SECONDS = 36 * 3600.0  # nightly cadence + one missed night of grace
DEFAULT_TAIL_BYTES = 64 * 1024
DEFAULT_HEALTH_INTERVAL_SECONDS = 300.0
_ESCALATION_KEY_PREFIX = "fork-sync-conflict-"
_ESCALATION_TITLE_PREFIX = "Fork-sync конфликт"
_OPEN_ESCALATION_SQL = (
    "SELECT COUNT(*) FROM tasks WHERE status NOT IN ('done', 'archived') "
    "AND (idempotency_key LIKE ? OR title LIKE ?)"
)

_LINE_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\]")
_RESULT_RE = re.compile(r"\bRESULT\s+(clean sync|created task|push failed|error)\b")
_RC_RE = re.compile(r"\(rc=(-?\d+)\)")
_TASK_ID_RE = re.compile(r"\bt_[0-9a-f]{6,}\b")

_RESULT_KIND = {
    "clean sync": "clean",
    "created task": "conflict",
    "push failed": "push_failed",
    "error": "error",
}
# Exit codes the hardened runner documents; used only when a legacy line carries no ``(rc=N)``.
_LEGACY_EXIT_CODES = {"clean": 0, "conflict": 2, "push_failed": 3, "error": 1}


@dataclass(frozen=True, slots=True)
class ForkSyncSettings:
    """Resolved ``fork_sync.*`` settings (env overrides win over config)."""

    enabled: bool = True
    log_path: Optional[Path] = None
    board: Optional[str] = None
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS
    tail_bytes: int = DEFAULT_TAIL_BYTES
    health_interval_seconds: float = DEFAULT_HEALTH_INTERVAL_SECONDS


@dataclass(frozen=True, slots=True)
class ForkSyncState:
    """One projection of the runner's artefacts into a bounded status."""

    status: str = UNKNOWN
    reason: str = "no_state"
    last_result: str = "unknown"
    last_exit_code: Optional[int] = None
    last_run_at: Optional[float] = None
    last_success_at: Optional[float] = None
    last_run_age_seconds: Optional[float] = None
    last_success_age_seconds: Optional[float] = None
    escalation_open: int = 0
    board_read_failed: bool = False

    @property
    def escalation_pending(self) -> bool:
        return self.escalation_open > 0

    def to_dict(self) -> dict[str, Any]:
        """Bounded, content-free projection for readiness/status surfaces."""
        return {
            "status": self.status,
            "reason": self.reason,
            "last_result": self.last_result,
            "last_exit_code": self.last_exit_code,
            "last_run_age_seconds": self.last_run_age_seconds,
            "last_success_age_seconds": self.last_success_age_seconds,
            "escalation_open": self.escalation_open,
            "board_read_failed": self.board_read_failed,
        }


@dataclass(frozen=True, slots=True)
class ForkSyncHealthSnapshot:
    """Projected state plus the bounded gauges it exports."""

    state: ForkSyncState
    metrics: list[GatewayMetric]

    @property
    def status(self) -> str:
        return self.state.status


@dataclass(frozen=True, slots=True)
class _LogScan:
    last_result: str = "unknown"
    last_exit_code: Optional[int] = None
    last_run_at: Optional[float] = None
    last_success_at: Optional[float] = None
    escalation_task_id: Optional[str] = None

    @property
    def has_state(self) -> bool:
        return self.last_result != "unknown"


def _positive_float(raw: Any, default: float, *, minimum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _positive_int(raw: Any, default: int, *, minimum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _default_log_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "logs" / "fork-sync.log"


def _config_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
    except Exception:
        logger.debug("fork-sync health: config unavailable", exc_info=True)
        return {}
    section = config.get("fork_sync") if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def resolve_fork_sync_settings(section: Optional[dict[str, Any]] = None) -> ForkSyncSettings:
    """Resolve settings from ``fork_sync.*`` config (loaded when *section* is None).

    ``FORK_SYNC_LOG`` — the runner's own env var — overrides the log path, so the
    feed and the script can never disagree about which file is the contract.
    """
    if section is None:
        section = _config_section()
    elif not isinstance(section, dict):
        section = {}

    log_path: Optional[Path] = None
    env_log = os.environ.get("FORK_SYNC_LOG", "").strip()
    if env_log:
        log_path = Path(env_log).expanduser()
    else:
        raw_path = str(section.get("log_path") or "").strip()
        log_path = Path(raw_path).expanduser() if raw_path else _default_log_path()

    return ForkSyncSettings(
        enabled=bool(section.get("enabled", True)),
        log_path=log_path,
        board=str(section.get("board") or "").strip() or None,
        stale_after_seconds=_positive_float(
            section.get("stale_after_seconds"), DEFAULT_STALE_AFTER_SECONDS, minimum=60.0
        ),
        tail_bytes=_positive_int(section.get("tail_bytes"), DEFAULT_TAIL_BYTES, minimum=4096),
        health_interval_seconds=_positive_float(
            section.get("health_interval_seconds"), DEFAULT_HEALTH_INTERVAL_SECONDS, minimum=30.0
        ),
    )


def _parse_line_ts(line: str) -> Optional[float]:
    match = _LINE_TS_RE.match(line)
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(1).rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=timezone.utc).timestamp()


def _line_message(line: str) -> str:
    match = _LINE_TS_RE.match(line)
    return line[match.end():].strip() if match else line.strip()


def _classify_line(message: str) -> Optional[str]:
    """Map one log line to an outcome, or None when the line is not an outcome.

    Ordered: the hardened ``RESULT`` contract first, then the legacy free-form
    lines the pre-hardening runner wrote.
    """
    match = _RESULT_RE.search(message)
    if match is not None:
        return _RESULT_KIND[match.group(1)]
    lowered = message.lower()
    if "rebase-sync done" in lowered or "nothing to sync" in lowered:
        return "clean"
    if lowered.startswith("conflict:") or "conflict task opened" in lowered:
        return "conflict"
    if "push failed" in lowered:
        return "push_failed"
    if lowered.startswith("error:") or lowered.startswith("fatal:"):
        return "error"
    return None


def _read_log_tail(path: Path, tail_bytes: int) -> str:
    with path.open("rb") as handle:
        try:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
        except OSError:
            handle.seek(0)
        return handle.read().decode("utf-8", errors="replace")


def scan_fork_sync_log(path: Optional[Path], *, tail_bytes: int = DEFAULT_TAIL_BYTES) -> _LogScan:
    """Parse the runner log tail.  A missing log is ``no_state``; any other read
    error propagates so the caller can report ``reader_error``."""
    if path is None:
        return _LogScan()
    try:
        text = _read_log_tail(path, tail_bytes)
    except FileNotFoundError:
        return _LogScan()

    last_result = "unknown"
    last_exit_code: Optional[int] = None
    last_run_at: Optional[float] = None
    last_success_at: Optional[float] = None
    escalation_task_id: Optional[str] = None

    for line in text.splitlines():
        message = _line_message(line)
        if not message:
            continue
        result = _classify_line(message)
        if result is None:
            continue
        stamp = _parse_line_ts(line)
        last_result = result
        if stamp is not None:
            last_run_at = stamp
            if result == "clean":
                last_success_at = stamp
        rc_match = _RC_RE.search(message)
        last_exit_code = int(rc_match.group(1)) if rc_match else _LEGACY_EXIT_CODES.get(result)
        if result == "conflict":
            task_match = _TASK_ID_RE.search(message)
            if task_match is not None:
                escalation_task_id = task_match.group(0)

    return _LogScan(
        last_result=last_result,
        last_exit_code=last_exit_code,
        last_run_at=last_run_at,
        last_success_at=last_success_at,
        escalation_task_id=escalation_task_id,
    )


def count_open_escalation_tasks(
    board: Optional[str] = None, *, db_path: Optional[Path] = None,
) -> int:
    """Read-only count of open fork-sync escalation tasks on *board*.

    Bounded and non-destructive (``mode=ro`` + ``query_only``), matching the
    readiness state-db probe.  Raises on an unreadable board — the caller
    reports that as a partial read, never as an exception.  ``db_path`` is the
    test seam (and the ``HERMES_KANBAN_DB`` override already exists for prod).
    """
    if db_path is None:
        from hermes_cli.kanban_db import kanban_db_path
        db_path = Path(kanban_db_path(board))
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=1.0)) as conn:
        conn.execute("PRAGMA query_only = ON")
        row = conn.execute(
            _OPEN_ESCALATION_SQL, (f"{_ESCALATION_KEY_PREFIX}%", f"{_ESCALATION_TITLE_PREFIX}%")
        ).fetchone()
    return max(0, int(row[0] or 0)) if row else 0


def classify_fork_sync_state(
    scan: _LogScan, *, escalation_open: int = 0, board_read_failed: bool = False,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS, now: Optional[float] = None,
) -> ForkSyncState:
    """Project one log scan (+ the board's escalation count) into a status."""
    now = time.time() if now is None else now
    run_age = None if scan.last_run_at is None else max(0.0, now - scan.last_run_at)
    success_age = None if scan.last_success_at is None else max(0.0, now - scan.last_success_at)

    def build(status: str, reason: str) -> ForkSyncState:
        return ForkSyncState(
            status=status, reason=reason, last_result=scan.last_result,
            last_exit_code=scan.last_exit_code, last_run_at=scan.last_run_at,
            last_success_at=scan.last_success_at, last_run_age_seconds=run_age,
            last_success_age_seconds=success_age, escalation_open=max(0, int(escalation_open)),
            board_read_failed=bool(board_read_failed),
        )

    if not scan.has_state:
        return build(UNKNOWN, "no_state")
    if scan.last_result == "clean":
        # Freshness is measured on the last SUCCESS, not the last run: a run that
        # synced nothing ("nothing to sync") still proves the fork is current.
        if success_age is not None and success_age > stale_after_seconds:
            return build(DEGRADED, "stale_sync")
        return build(HEALTHY, "clean_sync")
    if scan.last_result == "conflict":
        # A conflict leaves the fork diverged.  It is degraded (owned) while the
        # escalation task is open, unhealthy once nobody is looking at it.
        if board_read_failed or escalation_open > 0:
            return build(DEGRADED, "conflict_escalated")
        return build(UNHEALTHY, "conflict_unattended")
    if scan.last_result == "push_failed":
        return build(UNHEALTHY, "push_failed")
    return build(UNHEALTHY, "run_failed")


def read_fork_sync_state(settings: Optional[ForkSyncSettings] = None, *, now: Optional[float] = None) -> ForkSyncState:
    """Read both artefacts (independently fail-open) and project a status."""
    settings = settings or resolve_fork_sync_settings()

    scan = _LogScan()
    log_error: Optional[str] = None
    try:
        scan = scan_fork_sync_log(settings.log_path, tail_bytes=settings.tail_bytes)
    except Exception as exc:
        # Content-free: exception TYPE only (a message can carry a path).
        log_error = type(exc).__name__
        logger.warning("fork-sync health: log read failed (error_type=%s)", log_error)
        logger.debug("fork-sync health: log read traceback", exc_info=True)

    escalation_open = 0
    board_read_failed = False
    try:
        escalation_open = count_open_escalation_tasks(settings.board)
    except Exception as exc:
        board_read_failed = True
        logger.warning("fork-sync health: board read failed (error_type=%s)", type(exc).__name__)
        logger.debug("fork-sync health: board read traceback", exc_info=True)

    state = classify_fork_sync_state(
        scan, escalation_open=escalation_open, board_read_failed=board_read_failed,
        stale_after_seconds=settings.stale_after_seconds, now=now,
    )
    if log_error is not None and not scan.has_state:
        return ForkSyncState(
            status=UNKNOWN, reason="reader_error", last_result="unknown",
            escalation_open=state.escalation_open, board_read_failed=board_read_failed,
        )
    return state


def _metrics_for(state: ForkSyncState) -> list[GatewayMetric]:
    """Bounded gauges for the exported vocabulary (see ``_OBSERVABLE_METRIC_NAMES``)."""
    metrics = [
        GatewayMetric("hermes.fork_sync.up", 1 if state.status == HEALTHY else 0, {}),
        GatewayMetric("hermes.fork_sync.status", 1, {
            "hermes.fork_sync.status": _safe_metric_value(state.status),
            "hermes.fork_sync.reason": _safe_metric_value(state.reason),
        }),
        GatewayMetric(
            "hermes.fork_sync.last_exit_code",
            -1 if state.last_exit_code is None else int(state.last_exit_code),
            {},
        ),
        GatewayMetric("hermes.fork_sync.escalations_open", int(state.escalation_open), {}),
    ]
    if state.last_run_age_seconds is not None:
        metrics.append(GatewayMetric("hermes.fork_sync.last_run_age_seconds", float(state.last_run_age_seconds), {}))
    if state.last_success_age_seconds is not None:
        metrics.append(GatewayMetric(
            "hermes.fork_sync.last_success_age_seconds", float(state.last_success_age_seconds), {}
        ))
    return metrics


def build_fork_sync_health_snapshot(
    settings: Optional[ForkSyncSettings] = None, *, now: Optional[float] = None,
) -> ForkSyncHealthSnapshot:
    """Fresh read + projection.  Fail-open: a raising reader becomes a status."""
    settings = settings or resolve_fork_sync_settings()
    try:
        state = read_fork_sync_state(settings, now=now)
    except Exception as exc:
        logger.warning("fork-sync health: projection failed (error_type=%s)", type(exc).__name__)
        logger.debug("fork-sync health: projection traceback", exc_info=True)
        state = ForkSyncState(status=UNKNOWN, reason="reader_error")
    return ForkSyncHealthSnapshot(state=state, metrics=_metrics_for(state))


# ---- cached accessor + transition alerting ---------------------------------
_CACHE_TTL_SECONDS = 60.0
_cache_lock = threading.Lock()
_cached: Optional[tuple[float, ForkSyncHealthSnapshot]] = None


def _log_transition(previous: Optional[ForkSyncState], state: ForkSyncState) -> None:
    """Structured, content-free alert on a status change (deduped by the cache).

    WARNING+ records from ``gateway.*`` loggers are bridged to the monitoring
    plane by ``GatewayDiagnosticLogHandler``, so a degraded fork shows up in the
    operator's log sink as well as in the metrics.
    """
    if previous is not None and previous.status == state.status and previous.reason == state.reason:
        return
    fields = (
        "fork-sync health: status=%s reason=%s last_result=%s exit_code=%s "
        "last_success_age=%s escalations_open=%s board_read_failed=%s"
    )
    args = (
        state.status, state.reason, state.last_result, state.last_exit_code,
        None if state.last_success_age_seconds is None else round(state.last_success_age_seconds),
        state.escalation_open, state.board_read_failed,
    )
    if state.status in {DEGRADED, UNHEALTHY}:
        logger.warning("fork-sync health: status change — " + fields, *args)
    elif state.status == HEALTHY and previous is not None and previous.status != HEALTHY:
        logger.info("fork-sync health: recovered — " + fields, *args)


def refresh_fork_sync_health(
    settings: Optional[ForkSyncSettings] = None, *, now: Optional[float] = None,
) -> ForkSyncHealthSnapshot:
    """Force a refresh, update the cache and alert on a status transition."""
    global _cached
    with _cache_lock:
        previous = _cached[1].state if _cached is not None else None
    snapshot = build_fork_sync_health_snapshot(settings, now=now)
    with _cache_lock:
        _cached = (time.time(), snapshot)
    _log_transition(previous, snapshot.state)
    return snapshot


def fork_sync_health_snapshot(
    settings: Optional[ForkSyncSettings] = None, *, now: Optional[float] = None,
) -> ForkSyncHealthSnapshot:
    """Cached snapshot (TTL ``_CACHE_TTL_SECONDS``), refreshed fail-open on demand."""
    cached = _cached
    if cached is not None and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    return refresh_fork_sync_health(settings, now=now)


def reset_fork_sync_health_cache() -> None:
    """Drop the cache (tests, and after a config reload)."""
    with _cache_lock:
        global _cached
        _cached = None


__all__ = [
    "DEFAULT_HEALTH_INTERVAL_SECONDS", "DEFAULT_STALE_AFTER_SECONDS", "DEGRADED",
    "ForkSyncHealthSnapshot", "ForkSyncSettings", "ForkSyncState", "HEALTHY",
    "KNOWN_REASONS", "KNOWN_RESULTS", "KNOWN_STATUSES", "UNHEALTHY", "UNKNOWN",
    "build_fork_sync_health_snapshot", "classify_fork_sync_state",
    "count_open_escalation_tasks", "fork_sync_health_snapshot", "read_fork_sync_state",
    "refresh_fork_sync_health", "reset_fork_sync_health_cache", "resolve_fork_sync_settings",
    "scan_fork_sync_log",
]
