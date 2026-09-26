"""Kanban board watcher methods for GatewayRunner.

Background loops that subscribe to kanban boards, deliver notifications and
artifacts, and drive the multi-agent dispatcher. They use only ``self`` state,
so they live on a mixin ``GatewayRunner`` inherits. Per-tick work lives in
``kanban_watchers_notifier`` / ``kanban_watchers_dispatcher``; shared plumbing
in ``kanban_watchers_common``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _dispatcher_eligible_profiles,
    _dispatcher_holder_should_step_down,
    _dispatcher_holder_stealable,
    _dispatcher_lease_identity,
    _dispatcher_lease_profile,
    _dispatcher_profile_eligible,
    _dispatcher_takeover_challenge,
    _kanban_dispatch_allowed,
    _read_dispatcher_lease,
    _release_singleton_lock,
    _resolve_auto_decompose_settings,
    _write_dispatcher_lease,
    _gc_retention_days,
    _to_thread_process_service,
    logger,
)
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.kanban_watchers_dispatcher import (
    _KanbanDispatcher,
    _log_spawn_results,
    _resolve_dispatcher_settings,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_GC_INTERVAL_SECONDS = 3600.0
_HEALTH_WINDOW = 6

# Fork-sync health feed: settle before the first read so gateway startup never waits on it.
_FORK_SYNC_HEALTH_SETTLE_SECONDS = 10.0


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership before releasing the OS lock.

        Also truncates the lease record out of the lock file so a
        contender reading it after the release does not see a stale
        owner identity (the flock itself is what matters for exclusion,
        but a lingering record would mislead the takeover diagnostics).
        """
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        if handle is not None:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
            except OSError:
                pass
        _release_singleton_lock(handle)

    async def _sleep_between_ticks(self, interval: float) -> None:
        """Sleep *interval* (floored to 1s) in 1s slices so stop() never waits a full interval."""
        interval = max(interval, 1.0)
        slept = 0.0
        while slept < interval and self._running:
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        Per subscription, claims ``task_events`` newer than the stored cursor
        (kinds in TERMINAL_KINDS), sends one message per event, then advances
        the cursor. The subscription is removed only when the task is
        ``archived``: ``done`` is reversible, so the cursor — not unsubscribing
        — is the dedup mechanism (unsub-on-terminal dropped users when the
        dispatcher respawned a crashed task). All SQLite work runs in a thread;
        one tick's failure never stops the next.
        """
        try:
            from hermes_cli.config import load_config as _load_config

            cfg = _load_config()
            kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        except Exception as exc:
            logger.warning("kanban notifier: cannot load config (%s); continuing enabled", exc)
            kanban_cfg = {}
        if not kanban_cfg.get("notify_in_gateway", True):
            logger.info("kanban notifier: disabled via config kanban.notify_in_gateway=false")
            return

        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        sub_fail_counts: dict[tuple, int] = getattr(self, "_kanban_sub_fail_counts", {})
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name()
        self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        # Stale done-sub GC: subs survive ``done``, so boards that never
        # archive would accumulate rows scanned every tick. One DELETE per
        # board, at startup (0 → first tick) and at most hourly.
        _gc_next_at = 0.0

        while self._running:
            try:
                _gc_due = time.monotonic() >= _gc_next_at
                _retention = 30
                if _gc_due:
                    _gc_next_at = time.monotonic() + _GC_INTERVAL_SECONDS
                    _retention = _gc_retention_days()

                deliveries = await asyncio.to_thread(
                    _notifier_collect, self, _kb,
                    notifier_profile=notifier_profile, gc_due=_gc_due, gc_retention_days=_retention,
                )
                for d in deliveries:
                    await _KanbanNotification(
                        self, d, platform_cls=_Platform, sub_fail_counts=sub_fail_counts,
                    ).deliver()
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            await self._sleep_between_ticks(interval)

    def _kanban_sub_op(self, board: Optional[str], op: str, sub: dict, **extra: Any) -> None:
        """Sync helper (runs in to_thread): call ``kanban_db_notify.<op>`` for one subscription on its board."""
        from hermes_cli import kanban_db_connect as _kbc
        from hermes_cli import kanban_db_notify as _kbn
        conn = _kbc.connect(board=board)
        try:
            getattr(_kbn, op)(
                conn, task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "", **extra,
            )
        finally:
            conn.close()

    def _kanban_advance(self, sub: dict, cursor: int, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "advance_notify_cursor", sub, new_cursor=cursor)

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "remove_notify_sub", sub)

    def _kanban_rewind(self, sub: dict, claimed_cursor: int, old_cursor: int, board: Optional[str] = None) -> None:
        """Undo a claimed notification cursor after send failure."""
        self._kanban_sub_op(board, "rewind_notify_cursor", sub, claimed_cursor=claimed_cursor, old_cursor=old_cursor)

    async def _deliver_kanban_artifacts(self, *, adapter, chat_id: str, metadata: dict, event_payload: Optional[dict], task) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Sources, in priority order: ``event_payload['artifacts']``,
        ``event_payload['summary']``, then ``task.result`` (legacy). Paths are
        deduplicated, missing files are skipped (may be mentioned for
        reference only), and upload errors are logged, never raised.
        """
        raw_paths: list[str] = []
        prose_paths: list[str] = []
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                raw_paths += [item for item in raw if isinstance(item, str)]
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                prose_paths += adapter.extract_local_files(summary)[0]
        if task is not None and getattr(task, "result", None):
            prose_paths += adapter.extract_local_files(str(task.result))[0]
        # A staged copy and the scratch original it was copied from are the
        # same deliverable; on a review handoff the original still exists, so
        # prose mentions of it must not upload the file a second time.
        staged_names = {os.path.basename(p) for p in raw_paths}
        raw_paths += [p for p in prose_paths if os.path.basename(p) not in staged_names]
        candidates: list[str] = []
        for path in raw_paths:
            expanded = os.path.expanduser(path) if path else ""
            if expanded and expanded not in candidates and os.path.isfile(expanded):
                candidates.append(expanded)
        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        from urllib.parse import quote as _quote

        # Images ride one send_multiple_images call (batch uploads on Signal/Slack).
        image_paths = [p for p in candidates if Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if Path(p).suffix.lower() not in _IMAGE_EXTS]
        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(chat_id=chat_id, images=batch, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: image batch upload failed: %s", exc)
        for path in other_paths:
            try:
                if Path(path).suffix.lower() in _VIDEO_EXTS:
                    await adapter.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
                else:
                    await adapter.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: artifact upload (%s) failed: %s", path, exc)

    async def _kanban_dispatcher_boot(self) -> Optional[tuple]:
        """Resolve config, kanban_db and the singleton lock; None when the dispatcher must not run.

        Config is read once at boot (restart to apply), except the auto-decompose
        toggle which is re-read every tick. The env var is an escape hatch to
        disable without editing YAML. A contended singleton lock does not end the
        boot: this gateway stands by and retries the takeover on
        ``kanban.lock_takeover_interval`` (hence the coroutine), so a dead
        dispatcher-gateway is replaced instead of starving the board until the
        next container restart.
        """
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return None
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return None
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return None
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info("kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false")
            return None
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return None

        # Dispatch-eligibility gate (t_77f0d093): only profiles explicitly
        # allowed by kanban.dispatcher_lock_profiles (default ["default"]) may
        # even attempt the lock. Factory worker profiles (dev/lead/qa/reviewer),
        # helpers, and freshly created profiles must NOT race for it — a
        # misconfigured worker gateway that flips dispatch_in_gateway on
        # would otherwise steal the lock from the main gateway and starve
        # the board (incident 2026-08-13: helper held the lock for hours).
        _self_profile = _dispatcher_lease_profile()
        if not _dispatcher_profile_eligible(_self_profile, kanban_cfg):
            logger.info(
                "kanban dispatcher: profile %r is not a dispatch profile "
                "(kanban.dispatcher_lock_profiles=%r); this gateway will NOT "
                "dispatch and does not touch the singleton lock.",
                _self_profile, _dispatcher_eligible_profiles(kanban_cfg),
            )
            return None

        # Single-dispatcher backstop (see _acquire_singleton_lock). The lock
        # lives at the machine-global kanban root, so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        self._kanban_dispatcher_lock_path = _lock_path
        try:
            _takeover_interval = float(
                kanban_cfg.get("lock_takeover_interval", 30) or 30
            )
        except (TypeError, ValueError):
            _takeover_interval = 30.0
        _takeover_interval = max(_takeover_interval, 5.0)  # sanity floor
        try:
            _lease_timeout = float(kanban_cfg.get("lock_lease_timeout", 120) or 120)
        except (TypeError, ValueError):
            _lease_timeout = 120.0
        _lease_timeout = max(_lease_timeout, 10.0)

        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            # Durable takeover (t_fb4a7ca4, t_77f0d093): the flock is
            # released the moment the owner process dies, but a gateway
            # that found the lock contended at boot must not give up
            # forever. Re-check every kanban.lock_takeover_interval so a
            # dead dispatcher-gateway is replaced within ~a minute, and
            # use the lease record to detect a wedged owner or a
            # misconfigured non-dispatch holder.
            logger.info(
                "kanban dispatcher: another gateway holds the dispatcher "
                "lock (%s); standing by, will retry takeover every %.0fs "
                "(lease timeout %.0fs)",
                _lock_path, _takeover_interval, _lease_timeout,
            )
            _wedged_warned_at = 0.0
            while _lock_state == "contended" and self._running:
                _lease = _read_dispatcher_lease(_lock_path)
                _verdict = _dispatcher_holder_stealable(_lease, kanban_cfg)
                if _verdict == "dead":
                    logger.info(
                        "kanban dispatcher: lock holder pid=%s is dead; "
                        "retrying takeover",
                        _lease.get("pid"),
                    )
                elif _verdict == "non_factory":
                    logger.warning(
                        "kanban dispatcher: lock holder profile %r is not "
                        "allowed to dispatch; writing takeover challenge "
                        "and retrying",
                        _lease.get("profile"),
                    )
                    _dispatcher_takeover_challenge(
                        _lock_path, _verdict, _self_profile,
                    )
                elif _verdict == "stale":
                    _now = time.monotonic()
                    if _now - _wedged_warned_at >= 300:  # rate-limit
                        _wedged_warned_at = _now
                        logger.warning(
                            "kanban dispatcher: lock holder pid=%s has a "
                            "stale lease heartbeat (last %s); the dispatcher "
                            "loop is wedged but the process is alive, so the "
                            "flock cannot be stolen. Restart the holder "
                            "gateway, or it will recover on its own once the "
                            "process exits.",
                            _lease.get("pid"), _lease.get("heartbeat_at"),
                        )
                _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
                if _lock_state == "contended":
                    _slept = 0.0
                    while _slept < _takeover_interval and self._running:
                        await asyncio.sleep(min(0.5, _takeover_interval - _slept))
                        _slept += 0.5
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            self._kanban_dispatcher_lease_profile = _self_profile
            self._kanban_dispatcher_lease_started_at = int(time.time())
            _write_dispatcher_lease(_lock_handle, _dispatcher_lease_identity(self))
            logger.info(
                "kanban dispatcher: holding singleton dispatcher lock (%s) "
                "as profile %r",
                _lock_path, _self_profile,
            )
        elif _lock_state == "unavailable":
            logger.warning("kanban dispatcher: advisory lock unavailable at %s; proceeding "
                           "on config control alone.", _lock_path)
        else:
            # Still contended: the gateway is shutting down while waiting
            # for the takeover retry (self._running flipped False), or the
            # lock never became available. Do not fall through into the
            # dispatch loop without the lock.
            logger.info(
                "kanban dispatcher: lock %s still contended; not dispatching",
                _lock_path,
            )
            return None
        return _load_config, _kb, kanban_cfg

    async def _fork_sync_health_watcher(self) -> None:
        """Fork-sync divergence health feed — reports, never blocks dispatch.

        Refreshes :mod:`agent.monitoring.fork_sync_health` on an interval: age of
        the last successful fork/upstream sync, the last run's exit code, and how
        many conflict-escalation tasks are still open on the board. Every
        transition into degraded/unhealthy is logged as a structured WARNING
        (bridged to the monitoring plane for ``gateway.*`` loggers), and the
        state is readable from the readiness endpoint and the
        ``hermes.fork_sync.*`` gauges.

        Fail-open by construction: a broken log or board read is a *status*, not
        an exception, and the loop body is wrapped again here. The dispatcher and
        the ready queue must keep running while fork-sync is failed — that is the
        invariant this feed reports on, so it can never be the thing that breaks
        it.
        """
        from gateway.run_watchers import _interruptible_sleep

        try:
            from agent.monitoring import fork_sync_health
            settings = fork_sync_health.resolve_fork_sync_settings()
        except Exception as exc:
            logger.warning("fork-sync health: settings unavailable (error_type=%s)", type(exc).__name__)
            return
        if not settings.enabled:
            logger.info("fork-sync health: feed disabled via config fork_sync.enabled=false")
            return

        interval = max(30.0, float(settings.health_interval_seconds))
        # Short settle so gateway startup never waits on the feed.
        await asyncio.sleep(min(_FORK_SYNC_HEALTH_SETTLE_SECONDS, interval))
        while self._running:
            try:
                # Off-loop: the refresh reads a log file and the board DB.
                snapshot = await asyncio.to_thread(fork_sync_health.refresh_fork_sync_health, settings)
                self._fork_sync_health = snapshot.state
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Belt and braces: the feed is already fail-open, so a raise here means
                # a projection bug — still never a reason to stop dispatching.
                logger.warning("fork-sync health: refresh failed (error_type=%s)", type(exc).__name__)
                logger.debug("fork-sync health: refresh traceback", exc_info=True)
            await _interruptible_sleep(self, int(interval))

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` (default True); when false the
        loop exits and an external `hermes kanban daemon` is expected. Each
        tick runs :func:`kanban_db_dispatch.dispatch_once` in a thread; one tick's
        failure never stops the next. Shutdown: ``self._running`` is checked
        between ticks and the in-flight ``to_thread`` returns on its own.
        """
        boot = await self._kanban_dispatcher_boot()
        if boot is None:
            return
        _load_config, _kb, kanban_cfg = boot
        # Lock identity resolved at boot (argv profile + the machine-global lock
        # path); the per-tick holder recheck below reads them back off ``self``
        # instead of re-deriving them on every tick.
        _self_profile = getattr(self, "_kanban_dispatcher_lease_profile", "default")
        _lock_path = getattr(self, "_kanban_dispatcher_lock_path", "")
        settings = _resolve_dispatcher_settings(kanban_cfg, _kb)
        interval = settings.interval

        # Initial delay so adapters are wired before workers spawn (matches the notifier).
        await asyncio.sleep(5)

        # Health telemetry (mirrors `_cmd_daemon`): warn when the ready queue
        # is non-empty but spawns are 0 for N consecutive ticks — usually a
        # broken PATH, missing venv, or credential loss.
        bad_ticks = 0
        last_warn_at = 0
        results: Optional[list] = None
        dispatcher = _KanbanDispatcher(_kb, settings)

        logger.info("kanban dispatcher: embedded in gateway (interval=%.1fs)", interval)
        while self._running:
            # Holder-side periodic recheck (t_77f0d093): re-read the lease
            # every tick. If this gateway is no longer dispatch-eligible
            # (kanban.dispatch_in_gateway flipped off, or the profile was
            # removed from kanban.dispatcher_lock_profiles while running) or an
            # eligible contender challenged the lease (non_factory holder),
            # release the lock and stand down so the contender's takeover
            # retry can acquire it. This is the holder half of "restarting
            # a non-factory gateway no longer makes the main gateway lose
            # the lock": a misconfigured holder self-heals within one tick.
            if self._owns_kanban_dispatcher_lock():
                try:
                    _self_eligible = _dispatcher_profile_eligible(
                        _self_profile, kanban_cfg,
                    ) and bool(kanban_cfg.get("dispatch_in_gateway", True))
                    _holder_lease = _read_dispatcher_lease(_lock_path)
                    _challenge = _holder_lease.get("challenge") or {}
                    _challenger = (
                        _challenge.get("by") if isinstance(_challenge, dict) else None
                    )
                    _challenger_eligible = (
                        isinstance(_challenger, str)
                        and _dispatcher_profile_eligible(_challenger, kanban_cfg)
                    )
                    if _dispatcher_holder_should_step_down(
                        _holder_lease, _self_eligible, _challenger_eligible,
                    ):
                        _why = (
                            "profile no longer dispatch-eligible"
                            if not _self_eligible
                            else "takeover challenge from "
                            + str((_holder_lease.get("challenge") or {}).get("by"))
                        )
                        logger.warning(
                            "kanban dispatcher: %s; releasing singleton "
                            "dispatcher lock and standing down",
                            _why,
                        )
                        self._release_kanban_dispatcher_lock()
                        return
                    _write_dispatcher_lease(
                        self._kanban_dispatcher_lock_handle,
                        _dispatcher_lease_identity(self),
                    )
                except Exception:
                    logger.exception(
                        "kanban dispatcher: holder lease recheck failed",
                    )
            try:
                # Reap zombies before per-board work so a board DB failure
                # cannot block cleanup of unrelated workers.
                from hermes_cli import kanban_db_dispatch as _kbd
                pids = await _to_thread_process_service(_kbd.reap_worker_zombies)
                if pids:
                    logger.info("kanban dispatcher: reaped %d zombie worker(s), pids=%s", len(pids), pids)
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Emergency stop (`hermes pause`): no auto-decompose or
                # dispatch while paused; running workers finish naturally.
                if not _kanban_dispatch_allowed():
                    bad_ticks = 0
                else:
                    # Re-read the auto-decompose toggle live so disabling it
                    # takes effect on the next tick, not on restart.
                    _ad_enabled, _ad_per_tick = _resolve_auto_decompose_settings(_load_config)
                    # See #49638.
                    if _ad_enabled:
                        await _to_thread_process_service(dispatcher.auto_decompose_tick, _ad_per_tick)
                    results = await _to_thread_process_service(dispatcher.tick_once)
                    any_spawned = _log_spawn_results(results)
                    ready_pending = await _to_thread_process_service(dispatcher.ready_nonempty)
                    bad_ticks = bad_ticks + 1 if ready_pending and not any_spawned else 0
                now = int(time.time())
                if bad_ticks >= _HEALTH_WINDOW and now - last_warn_at >= 300:
                    held = _kbd.describe_suppression(res for _slug, res in (results or []))
                    logger.warning(
                        "kanban dispatcher stuck: ready queue non-empty for "
                        "%d consecutive ticks but 0 workers spawned.%s Check "
                        "profile health (venv, PATH, credentials) and "
                        "`hermes kanban list --status ready`.",
                        bad_ticks, f" Last tick held back: {held}." if held else "",
                    )
                    last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                self._release_kanban_dispatcher_lock()
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            await self._sleep_between_ticks(interval)

        self._release_kanban_dispatcher_lock()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
from contextvars import Context  # noqa: F401,E402
import logging  # noqa: F401,E402
import re  # noqa: F401,E402
import sqlite3  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    't': ('agent.i18n', 't'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
