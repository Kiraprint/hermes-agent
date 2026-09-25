"""Tests for the kanban dispatcher silent-stall escalation.

The incident this exists for: the factory board sat on one ready task held by
the respawn guard for ~21 h while ``kanban dispatcher stuck: ready queue
non-empty for N consecutive ticks`` grew in errors.log and nobody was told.
These cover the three signals (needs_attention, chat alert, self-healing
ticket) and, just as importantly, that a HEALTHY dispatcher stays silent.
"""

import asyncio
import importlib.util
import pathlib
import time

import pytest

# ``gateway.kanban_stuck_escalation`` is deliberately import-light (no
# gateway imports at module scope) so it can be unit-tested without booting
# the package. Load it by path, registering it in ``sys.modules`` FIRST —
# ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``
# and fails on a module that is not yet there.
_SPEC = importlib.util.spec_from_file_location(
    "gateway.kanban_stuck_escalation",
    pathlib.Path(__file__).resolve().parents[2] / "gateway" / "kanban_stuck_escalation.py",
)
_escalation = importlib.util.module_from_spec(_SPEC)
import sys as _sys

_sys.modules[_SPEC.name] = _escalation
_SPEC.loader.exec_module(_escalation)

PLATFORM_KEY = _escalation.PLATFORM_KEY
HoldTracker = _escalation.HoldTracker
StuckEscalationSettings = _escalation.StuckEscalationSettings
StuckEscalator = _escalation.StuckEscalator
clear_status = _escalation.clear_status
parse_alert_target = _escalation.parse_alert_target
resolve_settings = _escalation.resolve_settings
ticket_title = _escalation.ticket_title


def _settings(**overrides):
    base = dict(
        enabled=True,
        stalled_ticks=3,
        ready_age_seconds=600,
        alert_interval_seconds=7200,
        warn_age_seconds=60,
        auto_ticket=True,
        priority=1,
    )
    base.update(overrides)
    return StuckEscalationSettings(**base)


class _Clock:
    """Manually advanced monotonic clock — the escalator never moves it itself.

    Time is the thing these tests are about, so it advances only when a test
    says so. Left implicit (auto-ticking on read) every "held for 6 h" test
    silently becomes "held for 6 h and 3 calls", and the assertions stop
    meaning what they say.
    """

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds
        return self.t


def _make(clock=None, **overrides):
    """A StuckEscalator with recording collaborators and a manual clock."""
    events = {"status": [], "alerts": [], "tickets": []}
    clock = clock if clock is not None else _Clock()
    escalator = StuckEscalator(
        "factory",
        _settings(**overrides),
        write_status=lambda **kw: events["status"].append(kw),
        send_alert=_record(events["alerts"]),
        create_ticket=_record(events["tickets"]),
        clock=clock,
        monotonic=clock,
    )
    return escalator, events, clock


def _record(sink):
    """Recorder for the escalator's awaitable collaborators.

    ``send_alert`` is called as ``(evidence, text)`` and ``create_ticket`` as
    ``(payload)``, so the recorder takes ``*args`` and stores them all.
    """
    async def _inner(*args):
        sink.append(args[0] if len(args) == 1 else args)
    return _inner


class TestSettings:
    def test_defaults_when_config_unreadable(self):
        def _boom():
            raise RuntimeError("no config")

        # Observability must not be silenced by a config read failure.
        assert resolve_settings(_boom) == StuckEscalationSettings()

    def test_reads_live_block(self):
        cfg = {
            "kanban": {
                "stuck_escalation": {
                    "enabled": False,
                    "stalled_ticks": 7,
                    "ready_age_seconds": 900,
                    "alert_target": "telegram:477467153",
                }
            }
        }
        s = resolve_settings(lambda: cfg)
        assert s.enabled is False
        assert s.stalled_ticks == 7
        assert s.ready_age_seconds == 900
        assert s.alert_target == "telegram:477467153"

    def test_ticks_clamped_so_a_typo_cannot_alert_every_tick(self):
        cfg = {"kanban": {"stuck_escalation": {"stalled_ticks": 0}}}
        assert resolve_settings(lambda: cfg).stalled_ticks >= 1

    def test_non_dict_block_falls_back(self):
        assert resolve_settings(lambda: {"kanban": {"stuck_escalation": 5}}).stalled_ticks


class TestAlertTarget:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("telegram:477467153", ("telegram", "477467153", "")),
            ("telegram:-100123:12", ("telegram", "-100123", "12")),
            ("", None),
            ("telegram", None),
            ("telegram:477467153:1:2", None),
            (":477467153", None),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_alert_target(raw) == expected


class TestTicketTitle:
    def test_a_single_held_task_is_named_in_the_title(self):
        # The board is scanned by title, and the common case is exactly one
        # stuck task — its id is what the operator needs.
        assert "t_abc" in ticket_title("age", "factory", ["t_abc"])

    def test_several_held_tasks_are_summarized_by_count(self):
        title = ticket_title("ticks", "factory", ["a", "b", "c"])
        assert "3 tasks" in title
        assert "'factory'" in title and "ticks" in title

    def test_no_held_tasks_still_renders(self):
        assert ticket_title("ticks", "factory") == (
            "[auto] dispatcher stuck (ticks) on board 'factory' — no tasks"
        )


class TestHoldTracker:
    def test_hold_age_is_measured_from_the_current_hold(self):
        t = HoldTracker()
        assert t.update(["a"], 0) == {"a": 0.0}
        assert t.update(["a"], 90) == {"a": 90.0}

    def test_a_task_that_leaves_and_returns_restarts_its_clock(self):
        # A task that was spawned, finished and got re-queued is NOT "held for
        # 10 h" — otherwise a much later guard hold inherits an ancient age and
        # escalates instantly. It re-enters the held set at t=5010, so it has
        # been held for 0 s.
        t = HoldTracker()
        t.update(["a"], 0)
        t.update([], 5000)
        assert t.update(["a"], 5010) == {"a": 0.0}
        assert t.update(["a"], 5060) == {"a": 50.0}

    def test_a_newly_held_task_starts_its_clock_now(self):
        t = HoldTracker()
        t.update(["a"], 0)
        # "b" only entered the held set at t=100, so it has been held 0 s —
        # counting from t=0 would make a fresh hold look ancient.
        assert t.update(["a", "b"], 100) == {"a": 100.0, "b": 0.0}

    def test_oldest(self):
        t = HoldTracker()
        t.update(["a"], 0)
        t.update(["a", "b"], 50)
        assert t.oldest(["a", "b"], 60) == ("a", 60)


class TestEvaluate:
    def test_below_both_thresholds_does_not_escalate(self):
        esc, events, clock = _make(stalled_ticks=3, ready_age_seconds=600)
        assert esc.evaluate(["t1"], 2) is None
        clock.advance(120)
        assert esc.evaluate(["t1"], 2) is None
        assert events["status"] == [] and events["alerts"] == []

    def test_tick_threshold_escalates(self):
        esc, _events, _clock = _make(stalled_ticks=3, ready_age_seconds=600)
        ev = esc.evaluate(["t1"], 3)
        assert ev is not None
        assert ev["reason"] == "ticks"
        assert ev["oldest_task"] == "t1"
        assert ev["bad_ticks"] == 3

    def test_age_threshold_escalates_without_bad_ticks(self):
        # THE incident shape: a legitimate-looking hold that simply never
        # clears, with the tick counter nowhere near its threshold.
        esc, _events, clock = _make(stalled_ticks=1000, ready_age_seconds=600)
        esc.evaluate(["t1"], 0)
        clock.advance(21 * 3600)
        ev = esc.evaluate(["t1"], 0)
        assert ev is not None
        assert ev["reason"] == "age"
        assert ev["oldest_seconds"] == 21 * 3600

    def test_no_held_tasks_never_escalates(self):
        # Evidence gate: nothing held means the dispatcher is idle, not stuck.
        esc, events, _clock = _make(stalled_ticks=1)
        for _ in range(50):
            assert esc.evaluate([], 50) is None
        assert events["status"] == [] and events["tickets"] == []

    def test_disabled_is_a_total_no_op(self):
        esc, events, _clock = _make(enabled=False, stalled_ticks=1)
        assert esc.evaluate(["t1"], 999) is None
        assert esc.bad_ticks == 0
        assert events["alerts"] == [] and events["tickets"] == []

    def test_reason_is_both_when_two_signals_trip(self):
        esc, _events, clock = _make(stalled_ticks=3, ready_age_seconds=600)
        esc.evaluate(["t1"], 0)
        clock.advance(700)
        assert esc.evaluate(["t1"], 9)["reason"] == "both"

    def test_warns_at_the_softer_threshold_without_escalating(self, caplog):
        esc, events, clock = _make(
            stalled_ticks=1000, ready_age_seconds=10 ** 9, warn_age_seconds=60,
        )
        esc.evaluate(["t1"], 0)
        clock.advance(90)
        with caplog.at_level("WARNING"):
            assert esc.evaluate(["t1"], 0) is None
        # Operator tailing the log sees trouble brewing — but no alert, no
        # ticket, no status change.
        assert "held by respawn guard" in caplog.text
        assert events["status"] == [] and events["alerts"] == []
        assert events["tickets"] == []

    def test_a_task_joining_a_stalled_queue_is_not_immediately_old(self):
        # Two tasks held 6 h and 0 s respectively: the age signal must key off
        # the OLDEST one (6 h) and the evidence must carry both, but the fresh
        # task must not be reported as ancient.
        esc, _events, clock = _make(stalled_ticks=1000, ready_age_seconds=3600)
        esc.evaluate(["old"], 0)
        clock.advance(6 * 3600)
        ev = esc.evaluate(["old", "fresh"], 0)
        assert ev["oldest_task"] == "old"
        assert ev["holds"]["old"] == 6 * 3600
        assert ev["holds"]["fresh"] == 0


class TestEscalate:
    def test_emits_all_three_signals(self):
        esc, events, _clock = _make(stalled_ticks=3)
        ev = esc.evaluate(["t1"], 3)
        ev["guard_reasons"] = {"t1": "active_pr"}
        outcome = asyncio.run(esc.escalate(ev))
        assert outcome == {"status": "sent", "alert": "sent", "ticket": "created"}
        assert events["status"][0]["needs_attention"] is True
        assert events["status"][0]["platform"] == PLATFORM_KEY
        assert events["status"][0]["platform_state"] == "stalled"
        assert "t1" in events["status"][0]["error_message"]
        assert events["tickets"][0]["idempotency_key"] == ev["key"]
        # The ticket must name the guard reason, or the factory cannot act.
        assert "active_pr" in events["tickets"][0]["body"]

    def test_alert_text_carries_board_trigger_and_held_task(self):
        esc, events, _clock = _make(stalled_ticks=3)
        ev = esc.evaluate(["t1"], 3)
        asyncio.run(esc.escalate(ev))
        evidence, text = events["alerts"][0]
        assert "factory" in text
        assert ev["reason"] in text
        assert "t1" in text

    def test_second_tick_does_not_re_send_status_or_alert(self):
        esc, events, _clock = _make(stalled_ticks=3)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 3)))
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 4)))
        assert len(events["status"]) == 1
        assert len(events["alerts"]) == 1

    def test_ticket_is_retried_because_creation_is_idempotent(self):
        # Every tick re-requests the ticket; create_task's idempotency_key
        # collapses those into one card. Dropping the retry instead would lose
        # the card whenever the first attempt hit a locked board.
        esc, events, _clock = _make(stalled_ticks=3)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 3)))
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 4)))
        assert len(events["tickets"]) == 2
        assert (
            events["tickets"][0]["idempotency_key"]
            == events["tickets"][1]["idempotency_key"]
        )

    def test_alert_respects_the_repeat_interval(self):
        # A stall that outlives the interval gets a reminder — silence for
        # 21 h is the failure this whole change exists to prevent.
        # ready_age_seconds is kept huge so the trigger shape stays `ticks`
        # and the only variable under test is the elapsed time.
        esc, events, clock = _make(
            stalled_ticks=3, ready_age_seconds=10 ** 9, alert_interval_seconds=7200,
        )
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 3)))
        # Still inside the interval: no new alert even though the key is
        # already escalated.
        clock.advance(60)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 4)))
        assert len(events["alerts"]) == 1
        # Past the interval: the reminder fires.
        clock.advance(7200)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 5)))
        assert len(events["alerts"]) == 2
        # ...but the status latch and the idempotent ticket do not repeat.
        assert len(events["status"]) == 1
        assert len(events["tickets"]) == 3
        assert len({t["idempotency_key"] for t in events["tickets"]}) == 1

    def test_a_new_hold_gets_its_own_alert(self):
        # Different hold ⇒ different key ⇒ not suppressed by the previous
        # stall's dedup state.
        esc, events, _clock = _make(stalled_ticks=3)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 3)))
        asyncio.run(esc.escalate(esc.evaluate(["t2"], 4)))
        assert len(events["alerts"]) == 2

    def test_a_new_signal_shape_alerts_again(self):
        # The dedup key includes the trigger shape, so a stall that escalates
        # on `age` and later also trips the tick threshold raises a NEW alert
        # rather than hiding the worse state behind the first one. (A held task
        # only gets older, so the second shape is `both`, not `ticks`.)
        esc, events, clock = _make(
            stalled_ticks=2, ready_age_seconds=5, alert_interval_seconds=7200,
        )
        esc.evaluate(["t1"], 0)
        clock.advance(10)
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 0)))
        asyncio.run(esc.escalate(esc.evaluate(["t1"], 2)))
        assert [e[0]["reason"] for e in events["alerts"]] == ["age", "both"]

    def test_failing_alert_does_not_break_the_rest(self):
        sent = []

        async def _boom(evidence, text):
            raise RuntimeError("telegram down")

        esc = StuckEscalator(
            "factory", _settings(stalled_ticks=1),
            write_status=lambda **kw: sent.append(("status", kw)),
            send_alert=_boom,
            create_ticket=_record(sent),
            clock=time.time, monotonic=time.monotonic,
        )
        outcome = asyncio.run(esc.escalate(esc.evaluate(["t1"], 1)))
        assert outcome["alert"] == "skipped"
        assert outcome["status"] == "sent"
        assert any(isinstance(entry, tuple) and entry[0] == "status" for entry in sent)
        # The ticket is a plain payload, not a (kind, value) pair.
        assert any(isinstance(entry, dict) for entry in sent)

    def test_auto_ticket_can_be_disabled(self):
        esc, events, _clock = _make(stalled_ticks=3, auto_ticket=False)
        outcome = asyncio.run(esc.escalate(esc.evaluate(["t1"], 3)))
        assert outcome["ticket"] == "skipped"
        assert events["tickets"] == []

    def test_missing_collaborators_are_not_crashes(self):
        esc = StuckEscalator("factory", _settings(stalled_ticks=1))
        outcome = asyncio.run(esc.escalate(esc.evaluate(["t1"], 1)))
        assert outcome == {"status": "skipped", "alert": "skipped", "ticket": "skipped"}


class TestClearStatus:
    def test_clears_the_flag(self):
        calls = []
        clear_status(lambda **kw: calls.append(kw))
        assert calls == [{
            "platform": PLATFORM_KEY,
            "platform_state": "running",
            "error_code": None,
            "error_message": None,
            "needs_attention": False,
        }]

    def test_writer_failure_is_swallowed(self):
        def _boom(**kw):
            raise OSError("read-only fs")

        clear_status(_boom)  # must not raise


def test_ticket_title_names_the_trigger_and_board():
    assert "ticks" in ticket_title("ticks", "factory")


class TestSettingsRefresh:
    def test_replacing_settings_takes_effect_on_the_next_tick(self):
        # The dispatcher re-reads config every tick so `enabled: false` and
        # threshold edits apply without a gateway restart. It hands the live
        # settings object to an escalator that already exists, so evaluate()
        # must honour the replacement rather than the values captured at
        # construction.
        esc, events, _clock = _make(stalled_ticks=1000, ready_age_seconds=10 ** 9)
        assert esc.evaluate(["t1"], 5) is None
        esc.settings = _settings(stalled_ticks=3)
        assert esc.evaluate(["t1"], 3) is not None
        esc.settings = _settings(enabled=False)
        assert esc.evaluate(["t1"], 999) is None
        assert esc.bad_ticks == 0
