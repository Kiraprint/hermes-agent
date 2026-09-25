"""Escalation for a kanban dispatcher that stalls silently.

The dispatcher already warns in the log when its ready queue is non-empty but
it spawns nothing (``kanban dispatcher stuck: ready queue non-empty for N
consecutive ticks but 0 workers spawned``). That warning is only visible to
someone tailing ``errors.log``: an observed incident left the factory board
with a single ready task held by the respawn guard for ~21 h while the counter
climbed and nobody was told. This module turns that stall into three
operator-visible signals:

1. a ``needs_attention`` platform entry in ``gateway_state.json`` (surfaced by
   ``hermes status`` / the dashboard),
2. one chat alert per ``alert_interval_seconds`` — deduped, so a stall that
   lasts all night does not spam the operator,
3. an idempotent kanban ticket on the stalled board (keyed by guard reason +
   held task) so the factory can repair itself.

Design notes
------------

* **Two independent signals, OR-ed.** The consecutive bad-tick counter behind
  the existing warning, and how long the oldest held task has been waiting
  (measured from the current *unbroken* hold). A long "legitimate" hold
  therefore still escalates once it passes ``ready_age_seconds`` — that long
  hold is precisely what the incident was made of.
* **Evidence-gated.** Escalating without an identifiable held task would give
  the operator nothing to act on and the ticket nothing to carry, so an
  unexplained stall keeps the log warning as its only signal.
* **Alert budget.** The age signal fires once per hold (it is monotonic, so
  comparing it against a wall-clock interval would either spam or go silent);
  the tick signal is rate-limited by ``alert_interval_seconds``. Signal
  *shape* is also part of the key, so a stall that escalates on age and later
  starts tripping the tick counter alerts once more instead of hiding the
  newly-crossed threshold.
* **Nothing here touches the dispatcher.** Every side effect is injected by the
  caller (:class:`StuckEscalator` takes a status writer, a chat sender and a
  ticket creator), so this module stays import-light and testable without a
  gateway, and the dispatcher tick can never be broken by an escalation bug.
* **Long waits are logged long before they matter.** ``tick(held)`` emits a
  single DEBUG record the first time a hold crosses ``warn_age_seconds``
  (default 1 h), so an operator tailing the log sees trouble brewing well
  before the 6 h escalation threshold — without any side effect.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

logger = logging.getLogger("gateway.run")

#: Consecutive ticks with spawnable ready work and 0 spawns before escalating.
#: Same shape as the existing stuck warning, just loud instead of log-only.
DEFAULT_STALLED_TICKS = 20
#: How long one task may sit held before the age signal fires on its own.
#: 6 h is far beyond any legitimate review/spawn round-trip.
DEFAULT_READY_AGE_SECONDS = 6 * 60 * 60
#: Minimum gap between two chat alerts for the same stall.
DEFAULT_ALERT_INTERVAL_SECONDS = 2 * 60 * 60
#: Age at which a hold is logged (DEBUG, once per hold) without escalating.
DEFAULT_WARN_AGE_SECONDS = 60 * 60
#: Floors that keep a typo in config from turning the dispatcher into an
#: alert loop (``stalled_ticks: 0`` would escalate on every tick) or from
#: making the warn threshold impossible to reach.
_MIN_TICKS = 1
_MIN_SECONDS = 60


@dataclass(frozen=True)
class StuckEscalationSettings:
    """Resolved ``kanban.stuck_escalation`` config."""

    enabled: bool = True
    stalled_ticks: int = DEFAULT_STALLED_TICKS
    ready_age_seconds: int = DEFAULT_READY_AGE_SECONDS
    alert_interval_seconds: int = DEFAULT_ALERT_INTERVAL_SECONDS
    warn_age_seconds: int = DEFAULT_WARN_AGE_SECONDS
    #: ``"<platform>:<chat_id>[:<thread_id>]"`` — e.g. ``"telegram:477467153"``.
    #: Empty falls back to the connected home channel(s).
    alert_target: str = ""
    auto_ticket: bool = True
    #: Profile the self-healing ticket is assigned to. Empty leaves the ticket
    #: unassigned (the board's ``kanban.default_assignee`` still applies) — the
    #: safe default because this module has no idea which profiles exist.
    auto_ticket_assignee: str = ""
    priority: int = 1


def _coerce_int(raw: Any, default: int, minimum: int) -> int:
    """``raw`` as an int clamped to ``minimum``; unparsable values use ``default``."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def resolve_settings(load_config: Callable[[], Any]) -> StuckEscalationSettings:
    """Resolve the live ``kanban.stuck_escalation`` block, fail-safe.

    Read fresh on every dispatcher tick (same pattern as
    ``kanban.auto_decompose``) so ``enabled: false`` stops escalation on the
    next tick instead of requiring a gateway restart. A config read error or an
    unparsable sub-block returns the defaults: escalation is observability, and
    an unreadable config must neither crash the dispatcher nor silence it by
    accident.
    """
    try:
        cfg = load_config()
    except Exception:
        logger.debug("stuck escalation: config read failed; using defaults")
        return StuckEscalationSettings()
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    raw = kanban_cfg.get("stuck_escalation", {}) if isinstance(kanban_cfg, dict) else {}
    if not isinstance(raw, dict):
        return StuckEscalationSettings()
    return StuckEscalationSettings(
        enabled=bool(raw.get("enabled", True)),
        stalled_ticks=_coerce_int(raw.get("stalled_ticks"), DEFAULT_STALLED_TICKS, _MIN_TICKS),
        ready_age_seconds=_coerce_int(
            raw.get("ready_age_seconds"), DEFAULT_READY_AGE_SECONDS, _MIN_SECONDS,
        ),
        alert_interval_seconds=_coerce_int(
            raw.get("alert_interval_seconds"), DEFAULT_ALERT_INTERVAL_SECONDS, _MIN_SECONDS,
        ),
        warn_age_seconds=_coerce_int(
            raw.get("warn_age_seconds"), DEFAULT_WARN_AGE_SECONDS, _MIN_SECONDS,
        ),
        alert_target=str(raw.get("alert_target") or "").strip(),
        auto_ticket=bool(raw.get("auto_ticket", True)),
        auto_ticket_assignee=str(raw.get("auto_ticket_assignee") or "").strip(),
        priority=_coerce_int(raw.get("priority"), 1, 0),
    )


def parse_alert_target(target: str) -> Optional[tuple[str, str, str]]:
    """Split ``"<platform>:<chat_id>[:<thread_id>]"`` into its three parts.

    Returns ``None`` for an empty or malformed value so the caller falls back
    to the connected home channel(s) instead of sending the alert somewhere
    arbitrary.
    """
    parts = [p.strip() for p in str(target or "").split(":")]
    if len(parts) < 2 or len(parts) > 3:
        return None
    platform, chat_id = parts[0], parts[1]
    if not platform or not chat_id:
        return None
    return platform, chat_id, (parts[2] if len(parts) == 3 else "")


def _human_seconds(seconds: int) -> str:
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


class HoldTracker:
    """Tracks when each held ready task entered its current hold.

    The dispatcher is free to hold different tasks across ticks (``active_pr``
    flips between rows as PRs open and close), so the hold's *start* is the
    first tick a task is seen held, and it is only dropped when the task stops
    being held. That makes ``ready_age_seconds`` a measure of one unbroken
    hold — which is exactly the quantity the incident defined.
    """

    def __init__(self) -> None:
        self._start: dict[str, float] = {}

    def update(self, held: Sequence[str], now: float) -> dict[str, float]:
        """Return ``{task_id: seconds_held}`` for every ``held`` task.

        A task that entered a *new* hold (present in ``held`` but not in the
        tracker) is recorded at ``now``; a task that was already held keeps its
        original entry time so the hold is not reset on every stuck tick. Tasks
        no longer held are forgotten, so a later re-hold starts fresh.
        """
        current = set(held)
        for task_id in list(self._start):
            if task_id not in current:
                del self._start[task_id]
        for task_id in held:
            self._start.setdefault(task_id, now)
        return {tid: now - t0 for tid, t0 in self._start.items()}

    def oldest(self, held: Sequence[str], now: float) -> tuple[str, int]:
        """``(task_id, seconds_held)`` for the longest-held task, or ``("", 0)``."""
        holds = self.update(held, now)
        if not holds:
            return "", 0
        task_id, seconds = max(holds.items(), key=lambda kv: kv[1])
        return task_id, int(seconds)


# --------------------------------------------------------------------------
# Ticket body
# --------------------------------------------------------------------------


def ticket_title(
    reason: str, board: str, held: Sequence[str] = (),
) -> str:
    """Human-facing title of the self-healing ticket.

    Names the board, the trigger and how many tasks are stuck, so the board is
    scannable without opening the card. With a single held task its id is
    included too — that is the common case and the id is what an operator
    actually needs.
    """
    if len(held) == 1:
        scope = f"{held[0]}"
    elif held:
        scope = f"{len(held)} tasks"
    else:
        scope = "no tasks"
    return f"[auto] dispatcher stuck ({reason}) on board '{board}' — {scope}"


def ticket_body(
    *,
    board: str,
    reason: str,
    bad_ticks: int,
    settings: StuckEscalationSettings,
    held: Sequence[str],
    holds: dict[str, float],
    guard_reasons: dict[str, str],
) -> str:
    """Evidence an operator (or a dispatched worker) can act on."""
    window = int(max(1, settings.stalled_ticks) * 60)
    lines = [
        "Opened automatically by the kanban stuck-escalation watcher.",
        "",
        f"Board `{board}` had spawnable ready work for {bad_ticks} consecutive "
        f"dispatcher ticks and spawned nothing",
        f"(escalation threshold: {settings.stalled_ticks} ticks ≈ {window} min).",
        f"Trigger: {reason}.",
        "",
        "Held task(s) — skipped by the respawn guard:",
    ]
    for task_id in sorted(held):
        age = _human_seconds(int(holds.get(task_id, 0.0)))
        guard = guard_reasons.get(task_id) or "unknown"
        lines.append(f"- `{task_id}` — held {age} — guard reason `{guard}`")
    lines += [
        "",
        "Investigate before closing:",
        "1. `hermes kanban show <task_id>` — is the hold legitimate (PR waiting on",
        "   a human) or is it a stale guard (PR already merged/closed)?",
        "2. `hermes kanban tail` on the board for the `respawn_guarded` events.",
        "3. `hermes status` — the `kanban_dispatcher` entry stays",
        "   `needs_attention: true` until the dispatcher spawns again.",
        "",
        "This card is idempotent: the watcher will not open a second one while it",
        "stays `todo`/`ready`/`running` for the same board and trigger.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Escalator
# --------------------------------------------------------------------------

StatusWriter = Callable[..., None]
Sender = Callable[[dict[str, Any], str], Awaitable[Any]]
TicketCreator = Callable[[dict[str, Any]], Awaitable[Any]]


class StuckEscalator:
    """Escalates a stalled kanban dispatcher to the operator.

    Call :meth:`tick` once per dispatcher cycle, after the per-board results
    have been merged, with every task that is currently ``ready`` + assigned
    + unclaimed (``held``) and the reason the guard is holding back the worst
    one (``guard reason`` — ``"active_pr"``, ``"recent_success"``, …).
    """

    def __init__(
        self,
        board: str,
        settings: StuckEscalationSettings,
        *,
        write_status: Optional[StatusWriter] = None,
        send_alert: Optional[Sender] = None,
        create_ticket: Optional[TicketCreator] = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.board = board
        self.settings = settings
        self._write_status = write_status
        self._send_alert = send_alert
        self._create_ticket = create_ticket
        self._clock = clock
        self._monotonic = monotonic
        self._holds = HoldTracker()
        self._bad_ticks = 0
        self._alert_at: dict[str, float] = {}
        self._escalated_keys: set[str] = set()
        self._warned: set[str] = set()

    # ---- read-only state (also the test seam) ----

    @property
    def bad_ticks(self) -> int:
        return self._bad_ticks

    @property
    def needs_attention(self) -> bool:
        return self._bad_ticks > 0

    # ---- evaluation ----

    def evaluate(
        self, held: Iterable[str], bad_ticks: int,
    ) -> Optional[dict[str, Any]]:
        """Decide whether *this* tick escalates; returns the evidence dict or ``None``.

        Mutates only in-memory bookkeeping: the returned dict is what the
        caller feeds to :meth:`escalate`. Returns ``None`` when escalation is
        disabled, nothing is held (evidence gate), or no threshold is crossed.
        """
        held = list(held)
        if not self.settings.enabled:
            self._bad_ticks = 0
            return None
        if not held:
            # Nothing held: the dispatcher is idle or genuinely working. Forget
            # the holds (Done: any task that leaves the held set starts a fresh
            # clock) and clear the tick counter.
            self._holds.update([], self._monotonic())
            self._bad_ticks = 0
            return None

        self._bad_ticks = bad_ticks
        holds = self._holds.update(held, self._monotonic())
        oldest_id, oldest_age = ("", 0)
        if holds:
            oldest_id, oldest_age = max(holds.items(), key=lambda kv: kv[1])
            oldest_age = int(oldest_age)

        age_signal = oldest_age >= self.settings.ready_age_seconds
        tick_signal = bad_ticks >= self.settings.stalled_ticks

        if not (age_signal or tick_signal):
            # Long-but-not-yet-critical holds stay visible in the log only.
            if oldest_id and oldest_age >= self.settings.warn_age_seconds:
                key = f"{oldest_id}:{oldest_age // max(1, self.settings.warn_age_seconds)}"
                if key not in self._warned:
                    self._warned.add(key)
                    logger.warning(
                        "kanban dispatcher [%s]: %s held by respawn guard for %s "
                        "(escalates at %s)",
                        self.board, oldest_id, _human_seconds(oldest_age),
                        _human_seconds(self.settings.ready_age_seconds),
                    )
            return None

        if age_signal and tick_signal:
            reason = "both"
        elif age_signal:
            reason = "age"
        else:
            reason = "ticks"

        key = f"{self.board}:{oldest_id}:{reason}"
        already_escalated = key in self._escalated_keys
        if not already_escalated:
            self._escalated_keys.add(key)

        return {
            "reason": reason,
            "key": key,
            "bad_ticks": int(bad_ticks),
            "oldest_task": oldest_id,
            "oldest_seconds": oldest_age,
            "held": held,
            "holds": {tid: int(age) for tid, age in holds.items()},
            "already_escalated": already_escalated,
        }

    # ---- side effects (opt-in per collaborator) ----

    async def escalate(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Emit the status flag, the (rate-limited) alert and the ticket.

        Returns a dict of ``{signal: "sent"|"skipped"|"failed"}`` describing
        what actually happened, so callers/tests can assert on it and the
        dispatcher can log a single line.
        """
        outcome: dict[str, Any] = {"status": "skipped", "alert": "skipped", "ticket": "skipped"}
        if not evidence.get("already_escalated"):
            # needs_attention is a latch, not a pulse: written once per hold so
            # the runtime status flips on transition instead of on every tick.
            if await self._emit_status(evidence):
                outcome["status"] = "sent"
        # The alert is deliberately NOT gated on already_escalated. It carries
        # its own rate limit (alert_interval_seconds), so a stall that outlives
        # the interval gets a reminder. Gating it here too would mean silence
        # for the whole incident — the exact failure this exists to prevent.
        if await self._emit_alert(evidence):
            outcome["alert"] = "sent"
        if self.settings.auto_ticket:
            if await self._emit_ticket(evidence):
                outcome["ticket"] = "created"
        return outcome

    async def _emit_status(self, evidence: dict[str, Any]) -> bool:
        if self._write_status is None:
            return False
        try:
            writer = self._write_status
            if writer is None:
                # Imported lazily so this module can be unit-tested without a
                # full gateway package import.
                from gateway.status import write_runtime_status

                writer = write_runtime_status
            writer(
                platform=PLATFORM_KEY,
                platform_state="stalled",
                error_code="kanban_dispatcher_stalled",
                error_message=(
                    f"ready queue held by respawn guard for {evidence['bad_ticks']} "
                    f"consecutive ticks (threshold {self.settings.stalled_ticks}); "
                    f"held: {evidence['oldest_task'] or 'unknown'}"
                ),
                needs_attention=True,
            )
        except Exception:
            logger.exception("stuck escalation: runtime-status write failed")
            return False
        return True

    async def _emit_alert(self, evidence: dict[str, Any]) -> bool:
        if self._send_alert is None:
            return False
        now = self._monotonic()
        key = evidence["key"]
        last = self._alert_at.get(key)
        if last is not None and (now - last) < self.settings.alert_interval_seconds:
            return False
        self._alert_at[key] = now

        held = evidence["held"]
        holds = evidence["holds"]
        lines = [
            f"\u26a0\ufe0f factory stalled: dispatcher spawned nothing for "
            f"{evidence['bad_ticks']} ticks",
            f"board: {self.board}  |  trigger: {evidence['reason']}",
        ]
        for task_id in sorted(held)[:5]:
            lines.append(
                f"held: {task_id} — {_human_seconds(holds.get(task_id, 0))}"
            )
        if len(held) > 5:
            lines.append(f"...and {len(held) - 5} more held task(s)")
        lines.append("self-healing ticket opened on the board; needs_attention=true")
        try:
            await self._send_alert(evidence, "\n".join(lines))
        except Exception:
            logger.exception("stuck escalation: chat alert failed")
            return False
        return True

    async def _emit_ticket(self, evidence: dict[str, Any]) -> bool:
        if self._create_ticket is None:
            return False
        key = evidence["key"]
        payload = {
            "board": self.board,
            "title": ticket_title(evidence["reason"], self.board, evidence["held"]),
            "body": ticket_body(
                board=self.board,
                reason=evidence["reason"],
                bad_ticks=evidence["bad_ticks"],
                settings=self.settings,
                held=evidence["held"],
                holds=evidence["holds"],
                guard_reasons=evidence.get("guard_reasons") or {},
            ),
            # Same board + same worst held task + same trigger ⇒ one card, no
            # matter how many ticks the stall spans.
            "idempotency_key": key,
            "assignee": self.settings.auto_ticket_assignee,
            "priority": self.settings.priority,
        }
        try:
            await self._create_ticket(payload)
        except Exception:
            logger.exception("stuck escalation: auto-ticket creation failed")
            return False
        return True


#: Status-file key for the dispatcher's synthetic "platform" entry. Chosen to
#: read naturally in ``hermes status`` output next to real platforms.
PLATFORM_KEY = "kanban_dispatcher"


def clear_status(write_status: Optional[StatusWriter] = None) -> None:
    """Drop the ``needs_attention`` flag once the dispatcher spawns again.

    Called with no collaborators during a healthy tick; failures are logged at
    DEBUG because clearing is best-effort housekeeping and must never break the
    dispatcher loop.
    """
    try:
        writer = write_status
        if writer is None:
            from gateway.status import write_runtime_status

            writer = write_runtime_status
        writer(
            platform=PLATFORM_KEY,
            platform_state="running",
            error_code=None,
            error_message=None,
            needs_attention=False,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("stuck escalation: could not clear needs_attention", exc_info=True)
