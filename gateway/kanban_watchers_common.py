"""Plumbing shared by the kanban notifier and dispatcher loops.

Thread offload, board enumeration, live-config coercers and the singleton
dispatcher lock live here so the notifier, dispatcher and mixin modules read
them from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from contextvars import Context
from pathlib import Path
from typing import Any, Callable, Optional

# Keep the logger name run.py used so extracted log records are unchanged.
logger = logging.getLogger("gateway.run")


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the caller's context; a lingering
    ``delegate_task`` child marker would make ``write_txn`` false-trip for
    these process-owned writers. An empty Context keeps the DB guard intact
    for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work without inheriting request ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


def _list_boards(kb: Any) -> list:
    """Enumerate live boards; fall back to the default board when listing fails."""
    try:
        return kb.list_boards(include_archived=False)
    except Exception:
        return [kb.read_board_metadata(kb.DEFAULT_BOARD)]


def _board_slugs(kb: Any) -> list:
    return [b.get("slug") or kb.DEFAULT_BOARD for b in _list_boards(kb)]


def _positive_int_setting(kanban_cfg: dict, key: str) -> Optional[int]:
    """Parse an optional ``kanban.<key>`` int cap; None when unset or invalid (< 1 is invalid)."""
    raw = kanban_cfg.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.%s=%r; ignoring", key, raw)
        return None
    if value < 1:
        logger.warning("kanban dispatcher: kanban.%s=%r is below 1; ignoring", key, raw)
        return None
    logger.info("kanban dispatcher: %s=%d", key, value)
    return value


def _resolve_auto_decompose_settings(load_config: Callable[[], Any]) -> "tuple[bool, int]":
    """Live (enabled, per_tick) auto-decompose settings, re-read every dispatcher tick.

    Fails safe: a config read error returns ``(False, 3)`` rather than
    re-enabling a feature the user turned off. ``per_tick`` is clamped to ``>= 1``.

    Read fresh from config on every dispatcher tick (#49638) so that flipping ``kanban.auto_decompose:
    false`` to STOP runaway fan-out takes effect on the next tick instead of requiring a gateway restart.
    Auto-decompose is a safety toggle — a user who sees it create and launch tasks they didn't intend
    reaches for this flag to halt it, and a stale boot-captured value silently ignoring that change is the
    bug reported in #49638.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return bool(kcfg.get("auto_decompose", True)), max(per_tick, 1)


def _gc_retention_days() -> int:
    """``kanban.done_sub_retention_days`` (default 30; 0 disables), re-read per sweep; fails safe to 30."""
    try:
        from hermes_cli.config import load_config

        return int(((load_config() or {}).get("kanban") or {}).get("done_sub_retention_days", 30))
    except Exception:
        return 30


def _kanban_dispatch_allowed() -> bool:
    """False while the global emergency stop (`hermes pause`) is engaged.

    Checked every tick before spawning, so a pause applies on the next tick;
    in-flight workers are never touched. Fails open if estop is unimportable.
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take the exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway machine-wide may run the embedded dispatcher: concurrent
    dispatchers double reclaim frequency and claim events, and with
    ``wal_autocheckpoint=0`` concurrent manual checkpoints can corrupt index
    pages. ``dispatch_in_gateway`` is the primary control; this is the backstop.

    Returns ``(handle, "held")`` (release via :func:`_release_singleton_lock`),
    ``(None, "contended")`` when another process holds it (caller must NOT
    dispatch), or ``(None, "unavailable")`` when locking cannot be performed
    (caller falls back to config control).
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")  # windows-footgun: ok (append-mode lock handle, write not read)
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    with contextlib.suppress(Exception):
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    with contextlib.suppress(Exception):
        handle.close()

# ---------------------------------------------------------------------------
# Dispatcher lock lease (t_fb4a7ca4, t_77f0d093)
#
# The singleton flock alone is released automatically when the owner process
# dies, but nothing re-acquires it: every gateway checked the lock only at
# boot, so a dead dispatcher-gateway left the board without a dispatcher
# until the next container restart (incident 2026-08-13 15:13:44, board
# 'factory' starved for hours with 7 ready / 0 running). The lease record
# written INTO the lock file makes the holder observable (pid/profile/host/
# heartbeats) and lets contenders detect a dead or wedged owner, reclaim a
# stale lease, and force a misconfigured non-dispatch holder to stand down.
# ---------------------------------------------------------------------------
_DISPATCHER_LOCK_LEASE_VERSION = 1


def _dispatcher_lease_identity(self: Any) -> dict:
    """Return the identity fields of this gateway's lease record.

    Stored on the mixin at acquisition time so the per-tick heartbeat can
    rewrite the record without re-resolving the profile name.
    """
    return {
        "version": _DISPATCHER_LOCK_LEASE_VERSION,
        "profile": getattr(self, "_kanban_dispatcher_lease_profile", "default"),
        "pid": os.getpid(),
        "host": _dispatcher_hostname(),
        "started_at": getattr(self, "_kanban_dispatcher_lease_started_at", 0),
    }


def _dispatcher_hostname() -> str:
    try:
        import socket
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


def _dispatcher_lease_profile() -> str:
    """Best-effort name of the profile running this gateway.

    Lease records are diagnostics AND the input to the dispatch-eligibility
    gate, so a wrong label must never crash the dispatcher — every lookup
    path degrades gracefully. The final fallback is ``"default"``: a
    gateway launched without ``-p/--profile`` runs the default profile, and
    in the shared-HERMES_HOME deployment ``get_active_profile_name()``
    reports the same home for every gateway, so the argv flag is the only
    accurate per-gateway signal and its absence means default.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        name = get_active_profile_name()
        if name and name not in ("default", "custom"):
            return name
    except Exception:
        pass
    # Fall back to parsing our own argv for ``-p/--profile`` (the s6 run
    # scripts launch ``hermes -p <profile> gateway run``).
    _argv_profile = _dispatcher_profile_from_argv()
    if _argv_profile:
        return _argv_profile
    # No explicit profile flag: default profile gateway. A ``"custom"``
    # HERMES_HOME with no flag is the only ambiguity; defaulting to
    # "default" is safe because eligibility additionally requires
    # ``kanban.dispatch_in_gateway`` and membership in
    # ``kanban.dispatcher_lock_profiles`` (both defaulted for the default home).
    return "default"


def _dispatcher_profile_from_argv() -> Optional[str]:
    """``-p/--profile`` from our own argv, or None when this is not a hermes CLI run.

    Only a hermes entry point is consulted, and only a profile-shaped value is
    accepted. Other tools share the ``-p`` letter — pytest's ``-p <plugin>`` is
    the common one — and a bogus label is worse than no label here: the gate
    would refuse the lock and silently leave the board without a dispatcher.
    """
    try:
        argv = list(sys.argv or [])
        if not argv:
            return None
        entry = (argv[0] or "").replace("\\", "/").lower()
        base = entry.rsplit("/", 1)[-1]
        if not (base.startswith("hermes") or entry.endswith(("cli.py", "hermes_cli/main.py", "gateway/run.py"))):
            return None
        for i, arg in enumerate(argv[1:], start=1):
            if arg in ("-p", "--profile") and i + 1 < len(argv):
                candidate = str(argv[i + 1]).strip()
                if not candidate:
                    return None
                from hermes_constants import PROFILE_ID_RE
                return candidate if PROFILE_ID_RE.match(candidate) else None
    except Exception:
        return None
    return None


def _write_dispatcher_lease(handle, identity: dict) -> None:
    """Truncate-and-write the lease record (identity + fresh heartbeat)."""
    if handle is None:
        return
    record = dict(identity)
    record["heartbeat_at"] = int(time.time())
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(record))
        handle.flush()
        os.fsync(handle.fileno())
    except OSError:
        pass


def _read_dispatcher_lease(lock_path) -> dict:
    """Read the owner lease record from the lock file. Never raises."""
    try:
        with open(str(lock_path), "r", encoding="utf-8") as fh:
            data = fh.read()
        if not data.strip():
            return {}
        record = json.loads(data)
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def _lease_owner_alive(lease: dict) -> bool:
    """Best-effort liveness of the lease-recorded owner pid.

    Delegates to :func:`gateway.status._pid_exists`, the single cross-platform
    liveness probe: psutil first (zombies report dead, so a reaped-but-not-yet
    parented owner does not wedge the lease), then ctypes ``OpenProcess`` on
    Windows, then ``os.kill(pid, 0)`` on POSIX. Never call ``os.kill(pid, 0)``
    directly here — on Windows it is NOT a no-op (it sends CTRL_C_EVENT to the
    target's console process group, bpo-14484).
    """
    pid = lease.get("pid")
    # bool is a subclass of int — a lease record with pid=true is corrupt data,
    # not "pid 1 owns the lease", so reject it before it reaches the probe.
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists
    except ImportError:
        # gateway.status unavailable (partial install): on Windows assume alive
        # (the flock cannot be stolen from a live process anyway); elsewhere
        # fall through to a POSIX-only probe.
        if os.name == "nt":
            return True
        try:
            os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    return bool(_pid_exists(pid))


def _dispatcher_eligible_profiles(kanban_cfg) -> list:
    """Profiles allowed to run the singleton dispatcher (``kanban.dispatcher_lock_profiles``).

    Defaults to ``["default"]`` — only the default-profile gateway may hold
    the dispatcher lock unless a deployment explicitly opts other profiles
    in. This is the code-side half of the single-dispatcher posture: a
    worker/helper profile that accidentally flips ``dispatch_in_gateway``
    on (or inherits the default ``true``) must NOT race the main gateway
    for the lock.
    """
    raw = kanban_cfg.get("dispatcher_lock_profiles") if isinstance(kanban_cfg, dict) else None
    if raw is None:
        return ["default"]
    if isinstance(raw, str):
        raw = [raw]
    names = [str(p).strip() for p in raw if str(p).strip()]
    return names or ["default"]


def _dispatcher_profile_eligible(profile: str, kanban_cfg) -> bool:
    """Return whether *profile* may hold the singleton dispatcher lock."""
    if not profile or profile == "unknown":
        return False
    return profile in _dispatcher_eligible_profiles(kanban_cfg)


def _dispatcher_holder_stealable(lease: dict, kanban_cfg) -> Optional[str]:
    """Verdict on whether the recorded lock holder may be taken over.

    Returns one of:

    * ``"dead"`` — holder pid is gone; the flock was released by the kernel,
      so a retry acquire succeeds immediately.
    * ``"stale"`` — holder pid is alive but the heartbeat is older than
      ``kanban.lock_lease_timeout``; the dispatcher loop is wedged. A live
      flock cannot be stolen, so the contender warns and keeps retrying.
    * ``"non_factory"`` — holder pid is alive and heartbeating but the
      lease profile is not allowed to dispatch (``kanban.dispatcher_lock_profiles``
      + ``dispatch_in_gateway``); the holder is misconfigured and should
      stand down. The contender writes a challenge so the holder (running
      this code) self-releases on its next tick.
    * ``None`` — holder is healthy and entitled; do not touch it.

    Profile eligibility is judged BEFORE the heartbeat: a misconfigured
    non-dispatch holder must be challenged (so it steps down on its next
    tick) even when its heartbeat is stale — a stale non_factory holder
    would otherwise only ever be warned about.
    """
    if not lease:
        # No lease record: nothing observable to judge. The flock itself
        # decides — a contender simply retries the acquire.
        return None
    if not _lease_owner_alive(lease):
        return "dead"
    holder_profile = lease.get("profile") or ""
    if not _dispatcher_profile_eligible(holder_profile, kanban_cfg):
        return "non_factory"
    try:
        lease_timeout = float(kanban_cfg.get("lock_lease_timeout", 120) or 120)
    except (TypeError, ValueError):
        lease_timeout = 120.0
    lease_timeout = max(lease_timeout, 10.0)
    heartbeat_at = lease.get("heartbeat_at")
    if isinstance(heartbeat_at, (int, float)) and heartbeat_at > 0:
        if time.time() - heartbeat_at > lease_timeout:
            return "stale"
    return None


def _dispatcher_takeover_challenge(lock_path, verdict: str, challenger: str) -> None:
    """Write a takeover challenge into the lock file (best-effort).

    The challenger does not hold the flock, but the file is world-writable
    by design (it lives at the machine-global kanban root). The current
    holder re-reads the lease every tick and — when it sees a challenge
    from an eligible profile and/or discovers it is itself not eligible —
    releases the lock and stands down, letting the challenger's next retry
    acquire it. Pure advisory; a crash at any point just means the holder
    keeps running until the normal recovery paths (dead/stale) apply.
    """
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(lock_path), "r+", encoding="utf-8") as fh:
            data = fh.read()
            record = {}
            if data.strip():
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        record = parsed
                except ValueError:
                    record = {}
            record["challenge"] = {
                "by": challenger,
                "at": int(time.time()),
                "reason": verdict,
            }
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(record))
            fh.flush()
    except OSError:
        pass


def _dispatcher_holder_should_step_down(
    lease: dict, self_eligible: bool, challenger_eligible: bool = True,
) -> bool:
    """Return whether the current lock holder should voluntarily release.

    True when the holder's own profile is no longer dispatch-eligible
    (config flipped to ``dispatch_in_gateway: false``, or the profile was
    removed from ``kanban.dispatcher_lock_profiles`` while running) or when an
    eligible contender has written a takeover challenge naming a reason
    (``non_factory`` / ``stale`` / ``dead``). Step-down is cooperative: the
    holder releases the flock on its next tick so the contender's periodic
    recheck can acquire it — a live flock can never be stolen.

    A challenge from a profile that is ITSELF not allowed to dispatch is
    ignored: a rogue local process (or a misconfigured worker gateway)
    must not be able to bounce a healthy dispatcher. ``challenger_eligible``
    is computed by the caller from the challenge's ``by`` field against the
    live ``kanban.dispatcher_lock_profiles`` config.
    """
    if not self_eligible:
        return True
    challenge = lease.get("challenge") if isinstance(lease, dict) else None
    if isinstance(challenge, dict) and challenge.get("by"):
        return bool(challenger_eligible)
    return False


