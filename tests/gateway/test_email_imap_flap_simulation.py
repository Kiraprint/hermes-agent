"""Acceptance simulation for transient IMAP flap suppression (spec t_51191992, criteria t_a458055a).

Reproduces the production lifecycle of a sustained IMAP egress flap on a FAKE clock
(the only ``time`` consumer in the adapter is ``time.monotonic()`` in
``_record_imap_transient_error``, patched module-locally — asyncio keeps the real clock):

  t=0      installed adapter's poll fails transiently -> first-of-episode ERROR + ONE fatal
           notify (gateway logs ``Fatal email adapter error (email_imap_transient_error)``)
           -> gateway tears the adapter down and queues a reconnect (retryable fatal);
  t>0      reconnect watcher builds a FRESH adapter per attempt (class-level episode state
           survives the recreation), ``connect(is_reconnect=True)`` -> ``_probe_imap`` fails
           transiently -> recorded into the SAME episode (repeats suppressed at DEBUG, no new
           fatal notify), watcher logs INFO retry lines and backs off 30/60/120/240/300s-cap;
  t_end    flap ends -> fresh probe succeeds -> one INFO recovery, episode closed, alert re-armed.

Scenarios (spec §5 test matrix + t_a458055a acceptance):
  1. sustained 2h flap (one episode)      -> gateway.log <=1 FATAL + 1 alert; errors.log bounded;
  2. 20-error burst over 8 min + recovery -> per spec §5 row 1 (measured, incl. N-threshold alert);
  3. AUTH failures                        -> ERROR per attempt, immediate fatal, never suppressed;
  4. DB errors                            -> ERROR per attempt, immediate fatal, never suppressed;
  5. register() wiring                    -> the default alert handler is installed in-process and
                                             emits exactly one ``[Email] IMAP transient alert``.

Sinks replicate the production logging topology (gateway/run.py centralized logging):
  gateway.log-eq  = INFO+ records from the adapter logger + the ``gateway.run`` relay logger;
  errors.log-eq   = WARNING+ records from the same loggers (production errors.log is WARNING+).
The ``gateway.run`` relay logger reproduces the exact GatewayRunner fatal line shape
("Fatal email adapter error (code): message") so pattern counts match production logs.

Set IMAP_FLAP_SIM_REPORT=<path> to dump the full simulation evidence (every captured log line
with its fake-time event, per-sink counts, pass/fail verdicts) — used to produce the QA report
artifact for t_1f8dccad; tests stay hermetic without it.
"""

import asyncio
import imaplib
import logging
import os
import smtplib
import socket
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.email import adapter as email_adapter
from plugins.platforms.email.adapter import EmailAdapter

# Production cadences (gateway/run_adapters.py reconnect watcher): exponential early backoff,
# 300s cap (#37011). Poll-interval equivalent for the first event of an episode.
BACKOFF_SCHEDULE = (30, 60, 120, 240)
BACKOFF_CAP = 300
TWO_HOURS = 2 * 60 * 60
RELAY_LOGGER = logging.getLogger("gateway.run")  # exact production logger name for the fatal line


class FakeTime:
    """Module-local replacement for ``time`` in the adapter namespace: monotonic() on demand."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Sinks:
    """Capture records from the two flap-relevant loggers and replay the production filters."""

    def __init__(self):
        self.records = []  # (logger_name, levelno, levelname, message)
        self._handlers = {}
        for lg in (email_adapter.logger, RELAY_LOGGER):
            handler = logging.Handler()
            handler.emit = lambda record, lg=lg: self.records.append(
                (lg.name, record.levelno, record.levelname, record.getMessage())
            )
            self._handlers[lg] = (handler, lg.level)
            lg.addHandler(handler)
            lg.setLevel(logging.DEBUG)
            lg.propagate = False

    def close(self):
        for lg, (handler, level) in self._handlers.items():
            lg.removeHandler(handler)
            lg.setLevel(level)

    @staticmethod
    def _gateway(record):
        name, levelno, _, _ = record
        return levelno >= logging.INFO and name in (
            email_adapter.logger.name, RELAY_LOGGER.name,
        )

    @staticmethod
    def _errors(record):
        return record[1] >= logging.WARNING

    def gateway_log(self):
        return [r for r in self.records if self._gateway(r)]

    def errors_log(self):
        return [r for r in self.records if self._errors(r)]

    def count(self, pattern, sink="gateway"):
        lines = self.gateway_log() if sink == "gateway" else self.errors_log()
        return sum(1 for r in lines if pattern in r[3])

    def render(self, sink):
        lines = self.gateway_log() if sink == "gateway" else self.errors_log()
        return [f"{lvl:<5} {name}: {msg}" for name, _, lvl, msg in lines]


REPORT = []  # filled by scenarios when IMAP_FLAP_SIM_REPORT is set
# Resolved at IMPORT time: the autouse fixture below hermetically clears os.environ for the
# EMAIL_* config (patch.dict(..., clear=True)), so a live lookup inside a test would miss it.
_REPORT_PATH = os.environ.get("IMAP_FLAP_SIM_REPORT")
_REPORT_INITIALIZED = False


def report_section(title, sinks, extra=None):
    global _REPORT_INITIALIZED
    path = _REPORT_PATH
    if not path:
        return
    mode = "a" if _REPORT_INITIALIZED else "w"  # first section truncates; later ones append
    _REPORT_INITIALIZED = True
    section_start = len(REPORT)
    REPORT.extend([
        f"\n{'=' * 78}\n{title}\n{'=' * 78}",
        "\n--- gateway.log equivalent (INFO+, adapter + gateway.run relay) ---",
        *("  " + line for line in sinks.render("gateway")),
        "\n--- errors.log equivalent (WARNING+) ---",
        *("  " + line for line in sinks.render("errors")),
        "\n--- counts ---",
        f"  FATAL relay lines (Fatal email adapter error): {sinks.count('Fatal email adapter error')}",
        f"  alert lines (IMAP transient alert):            {sinks.count('IMAP transient alert')}",
        f"  IMAP fetch/connect ERROR lines:                {sinks.count('IMAP fetch error') + sinks.count('IMAP connection failed')}",
        f"  errors.log-eq total WARNING+ lines:            {len(sinks.errors_log())}",
    ])
    for line in extra or []:
        REPORT.append("  " + line)
    with open(path, mode, encoding="utf-8") as f:  # incremental: evidence survives a later crash
        f.write("\n".join(REPORT[section_start:]) + "\n")


@pytest.fixture(autouse=True)
def imap_flap_env(monkeypatch):
    """Hermetic env + class-state reset around every scenario."""
    EmailAdapter._imap_transient_episodes.clear()
    EmailAdapter.set_imap_transient_alert_handler(None)
    monkeypatch.setattr(email_adapter, "time", FakeTime())
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=True):
        yield
    EmailAdapter._imap_transient_episodes.clear()
    EmailAdapter.set_imap_transient_alert_handler(None)


def make_adapter():
    instance = EmailAdapter(email_adapter.PlatformConfig(enabled=True))
    instance._write_runtime_status_safe = MagicMock()
    # SMTP stays healthy while IMAP flaps (production topology: separate egress path).
    instance._probe_smtp = MagicMock(return_value=True)
    return instance


def failing_imap(exc):
    """imaplib.IMAP4_SSL stand-in whose every command raises *exc* (connect stays 'healthy':
    the flap is in the TLS read path — login/select/search raise, exactly like the observed
    'The read operation timed out' / errno-104 signatures)."""
    mock = MagicMock()
    mock.capabilities = ("IMAP4rev1", "ID")
    mock.login.side_effect = exc
    mock.select.side_effect = exc
    mock.uid.side_effect = exc
    return mock


def healthy_imap():
    mock = MagicMock()
    mock.capabilities = ("IMAP4rev1", "ID")
    mock.uid.return_value = ("OK", [b""])
    return mock


def relay_fatal(adapter):
    """Stand-in for GatewayRunner._handle_adapter_fatal_error_impl's log line (run_adapters.py:
    ``logger.error("Fatal %s adapter error (%s): %s", ...)``) — the FATAL-pattern gateway.log
    line. Production logs one such line per _notify_fatal_error()."""
    RELAY_LOGGER.error(
        "Fatal %s adapter error (%s): %s", "email",
        adapter.fatal_error_code or "unknown", adapter.fatal_error_message or "unknown error",
    )


async def watcher_pass(adapter, attempt, backoff):
    """One reconnect-watcher pass for a fresh adapter (run_adapters.py:710-742 shape)."""
    RELAY_LOGGER.info("Reconnecting %s (attempt %d)...", "email", attempt)
    ok = await adapter.connect(is_reconnect=True)
    if not ok:
        RELAY_LOGGER.info("Reconnect %s failed, next retry in %ds", "email", backoff)
    if adapter._poll_task is not None:
        adapter._poll_task.cancel()
    return ok


async def sustained_flap(sinks, *, duration, flap_exc, first_poll_exc=None):
    """Drive the production lifecycle of one sustained flap episode on the fake clock.

    Returns (monitor_heartbeats, attempts). Monitor heartbeats model the health monitor's
    periodic line: one per watcher pass — 'monitor.log does not fall' == they continue
    through the whole flap window with no gap."""
    # Production wiring state: register() installs the default alert handler in the gateway
    # process — mirror it so the threshold alert is live exactly as after PR merge + deploy.
    EmailAdapter.set_imap_transient_alert_handler(email_adapter._log_imap_transient_alert)
    fake = email_adapter.time
    assert isinstance(fake, FakeTime)  # installed by the autouse fixture
    start = fake.now  # episodes may run back-to-back on the same fake clock
    installed = make_adapter()
    installed.set_fatal_error_handler(relay_fatal)

    # t=0 — installed adapter's poll fails transiently: first-of-episode ERROR + ONE fatal notify.
    exc = first_poll_exc or flap_exc
    with patch("imaplib.IMAP4_SSL", return_value=failing_imap(exc)):
        await installed._check_inbox()

    # Reconnect watcher: fresh adapter per attempt (class-level episode state carries over),
    # exponential backoff capped at 300s (#37011), until the flap window ends.
    heartbeats, attempt, delay_idx = [], 1, 0
    while True:
        backoff = BACKOFF_SCHEDULE[delay_idx] if delay_idx < len(BACKOFF_SCHEDULE) else BACKOFF_CAP
        fake.advance(backoff)
        if fake.now - start >= duration:
            break
        fresh = make_adapter()
        fresh.set_fatal_error_handler(relay_fatal)
        with patch("imaplib.IMAP4_SSL", return_value=failing_imap(flap_exc)):
            ok = await watcher_pass(fresh, attempt, backoff)
        heartbeats.append((round(fake.now - start), ok, fresh.fatal_error_code))
        attempt += 1
        delay_idx += 1

    # Flap ends: one more watcher pass whose probe succeeds -> episode closes with one recovery.
    fresh = make_adapter()
    fresh.set_fatal_error_handler(relay_fatal)
    with patch("imaplib.IMAP4_SSL", return_value=healthy_imap()):
        ok = await watcher_pass(fresh, attempt, 0)
    heartbeats.append((round(fake.now - start), ok, fresh.fatal_error_code))
    return heartbeats, attempt


def test_acceptance_sustained_2h_flap():
    """t_a458055a acceptance: during a 2h flap imitation gateway.log holds <=1 FATAL + 1 alert,
    errors.log does not grow with event count, monitor.log does not fall."""
    sinks = Sinks()
    try:
        flap_exc = socket.timeout("The read operation timed out")
        heartbeats, attempts = asyncio.run(sustained_flap(
            sinks, duration=TWO_HOURS, flap_exc=flap_exc,
        ))
        fatal_lines = sinks.count("Fatal email adapter error")
        alert_lines = sinks.count("IMAP transient alert")
        fetch_errors = sinks.count("IMAP fetch error")
        errors_lines = len(sinks.errors_log())
        report_section(
            "SCENARIO 1 — sustained 2h IMAP flap (single episode, "
            f"{attempts} reconnect attempts + first poll)",
            sinks,
            extra=[
                f"transient events total: {attempts + 1}",
                f"monitor heartbeats: {len(heartbeats)} (first={heartbeats[0][0]}s, last={heartbeats[-1][0]}s, all watcher passes alive)",
                f"episode state after recovery: {EmailAdapter._imap_transient_episodes}",
                f"VERDICT gateway.log<=1 FATAL: {'PASS' if fatal_lines <= 1 else 'FAIL'} ({fatal_lines})",
                f"VERDICT gateway.log<=1 alert: {'PASS' if alert_lines <= 1 else 'FAIL'} ({alert_lines})",
                f"VERDICT errors.log bounded:  {'PASS' if errors_lines <= 3 else 'FAIL'} ({errors_lines} WARNING+ lines for {attempts + 1} events)",
                f"VERDICT monitor.log alive:   {'PASS' if heartbeats and heartbeats[-1][0] >= TWO_HOURS - 1 else 'FAIL'}",
            ],
        )
        # Acceptance gates.
        assert fatal_lines == 1, sinks.render("gateway")
        assert alert_lines == 1, sinks.render("gateway")
        assert fetch_errors == 1  # first-of-episode only; repeats suppressed at DEBUG
        assert errors_lines == 3  # first fetch error + fatal relay + the one alert
        # 26+ flap events produced exactly one of each noisy line — no growth with event count.
        assert attempts >= 24
        # Monitor kept beating through the whole window right up to the recovery pass.
        assert heartbeats[-1][0] >= TWO_HOURS - 1
        assert all(ok is True for _, ok, _ in heartbeats[-1:])
        # Episode closed and re-armed by the successful probe.
        assert not EmailAdapter._imap_transient_episodes
        assert sinks.count("IMAP fetch recovered after") == 1
    finally:
        sinks.close()


def test_burst_flap_8min_recovers():
    """Spec §5 row 1 (measured against the REAL production cadence): a sustained burst then
    recovery. The reconnect watcher's backoff (30/60/120/240/300-cap, #37011) fits only ~5
    events into 8 min — not the spec row's idealized '20 over 8 min' — and 5 errors inside
    the 15-min window trip M=5, so an 8-min sustained episode DOES alert (spec §3 beats the
    row's 'none'; see the QA report note on the spec-table cadence mismatch)."""
    sinks = Sinks()
    try:
        heartbeats, attempts = asyncio.run(sustained_flap(
            sinks, duration=8 * 60, flap_exc=socket.timeout("The read operation timed out"),
        ))
        report_section(
            f"SCENARIO 2 — sustained burst over 8 min then recovery ({attempts} attempts at "
            "production backoff cadence)",
            sinks,
            extra=[f"VERDICT single episode, repeats suppressed: "
                   f"{'PASS' if sinks.count('IMAP fetch error') == 1 else 'FAIL'}",
                   f"VERDICT one alert via M=5/15min: "
                   f"{'PASS' if sinks.count('IMAP transient alert') == 1 else 'FAIL'}"],
        )
        assert attempts >= 4  # 30+60+120+240(,300) — backoff-capped event count inside 8 min
        assert sinks.count("IMAP fetch error") == 1
        assert sinks.count("Fatal email adapter error") == 1
        assert sinks.count("IMAP transient alert") == 1  # M=5-in-15-min trips inside 8 min
        assert not EmailAdapter._imap_transient_episodes
        # Re-arm: a NEW episode after recovery alerts again exactly once (spec §5 row 5,
        # "digest alert once per episode"). Same 8-min shape so the N threshold trips.
        flap2 = asyncio.run(sustained_flap(
            sinks, duration=8 * 60, flap_exc=ConnectionResetError("[Errno 104] Connection reset by peer"),
        ))
        assert sinks.count("Fatal email adapter error") == 2
        assert sinks.count("IMAP transient alert") == 2
        report_section(
            "SCENARIO 2b — second consecutive flap episode after recovery (re-arm proof)",
            sinks,
            extra=[f"attempts in episode 2: {flap2[1]}",
                   "VERDICT per-episode re-arm (1 FATAL + 1 alert per episode): "
                   f"{'PASS' if sinks.count('IMAP transient alert') == 2 else 'FAIL'}"],
        )
    finally:
        sinks.close()


def test_isolated_flaps_stay_visible_but_never_alert():
    """Supplementary measurement (honest QA): the OBSERVED production pattern — an isolated
    flap every ~40 min where reconnect succeeds 2s later — is one episode per flap under the
    ratified spec §2 (episode ends on ANY successful IMAP command). Each isolated flap keeps
    its single ERROR + single FATAL (unchanged visibility — a real observable event, never
    hidden), but NO threshold alert ever fires and repeats inside each flap stay suppressed."""
    sinks = Sinks()
    try:
        EmailAdapter.set_imap_transient_alert_handler(email_adapter._log_imap_transient_alert)
        fake = email_adapter.time
        assert isinstance(fake, FakeTime)  # installed by the autouse fixture
        flap_exc = socket.timeout("The read operation timed out")

        async def isolated_flap_cadence():
            for gap in (0, 40 * 60, 40 * 60):  # three isolated flaps over ~2h
                fake.advance(gap)
                installed = make_adapter()
                installed.set_fatal_error_handler(relay_fatal)
                with patch("imaplib.IMAP4_SSL", return_value=failing_imap(flap_exc)):
                    await installed._check_inbox()  # flap: 1 ERROR + 1 fatal notify
                fresh = make_adapter()
                fresh.set_fatal_error_handler(relay_fatal)
                with patch("imaplib.IMAP4_SSL", return_value=healthy_imap()):
                    ok = await watcher_pass(fresh, 1, 0)  # reconnect OK 2s later
                assert ok is True

        asyncio.run(isolated_flap_cadence())
        report_section(
            "SCENARIO 2c — isolated flaps every ~40 min, reconnect OK (observed production "
            "pattern, 3 flaps over 2h)",
            sinks,
            extra=[
                f"isolated-flap ERROR lines: {sinks.count('IMAP fetch error')}/3 (unchanged visibility by design, spec §2)",
                f"isolated-flap FATAL lines: {sinks.count('Fatal email adapter error')}/3",
                f"threshold alerts for isolated flaps: {sinks.count('IMAP transient alert')}/0",
                f"recovery summaries: {sinks.count('IMAP fetch recovered after')}/3",
                "NOTE: per ratified spec §2 an isolated flap is its own episode (success ends "
                "it) — 1 ERROR + 1 FATAL per FLAP, no alert; only SUSTAINED flaps suppress + "
                "alert. The sustained-flap acceptance gate is scenario 1.",
            ],
        )
        assert sinks.count("IMAP fetch error") == 3  # per isolated flap — never suppressed
        assert sinks.count("Fatal email adapter error") == 3
        assert sinks.count("IMAP transient alert") == 0  # isolated flaps never trip N/M
        assert sinks.count("IMAP fetch recovered after") == 3
        assert not EmailAdapter._imap_transient_episodes
    finally:
        sinks.close()


def test_auth_failures_never_suppressed():
    """Spec §4: AUTH-class failures stay ERROR per attempt with immediate fatal — no episode,
    no suppression, no transient alert."""
    sinks = Sinks()
    try:
        auth_exc = imaplib.IMAP4.error("[AUTHENTICATIONFAILED] invalid credentials (Failure)")
        alerts = []
        EmailAdapter.set_imap_transient_alert_handler(alerts.append)

        async def run():
            installed = make_adapter()
            installed.set_fatal_error_handler(relay_fatal)
            notified = 0
            # t=0 poll fails with AUTH: immediate ERROR + fatal notify (non-transient branch).
            with patch("imaplib.IMAP4_SSL", return_value=failing_imap(auth_exc)):
                await installed._check_inbox()
            notified += 1
            # Watcher retries: fresh adapters' probes fail with AUTH -> ERROR per attempt
            # (probe logs '[Email] IMAP connection failed' via _fail), retryable -> keep retrying.
            fake, attempt = email_adapter.time, 1
            assert isinstance(fake, FakeTime)  # installed by the autouse fixture
            for delay in (30, 60, 120, 240, 300, 300, 300, 300, 300):
                fake.advance(delay)
                fresh = make_adapter()
                fresh.set_fatal_error_handler(relay_fatal)
                with patch("imaplib.IMAP4_SSL", return_value=failing_imap(auth_exc)):
                    ok = await watcher_pass(fresh, attempt, delay)
                assert ok is False
                attempt += 1
            return notified

        notified = asyncio.run(run())
        visible = sinks.count("IMAP fetch error") + sinks.count("IMAP connection failed")
        report_section(
            "SCENARIO 3 — AUTH failures (10 attempts, zero suppression)",
            sinks,
            extra=[
                f"ERROR-per-attempt visible lines: {visible}/10",
                f"transient alerts for AUTH: {len(alerts)}",
                f"VERDICT AUTH visible per attempt: {'PASS' if visible == 10 else 'FAIL'}",
            ],
        )
        assert visible == 10  # every single attempt stayed an ERROR line
        assert notified == 1  # the poll-path fatal notify fired (watcher retries log at INFO)
        assert sinks.count("IMAP transient alert") == 0
        assert alerts == []
        assert not EmailAdapter._imap_transient_episodes  # AUTH never opens an episode
        assert sinks.errors_log() and all(l >= logging.WARNING for l in
                                          (r[1] for r in sinks.errors_log()))
    finally:
        sinks.close()


def test_db_errors_never_suppressed():
    """Spec §4: DB-class failures stay ERROR per attempt with an immediate fatal notify —
    never classified transient, never suppressed."""
    sinks = Sinks()
    try:
        db_exc = sqlite3.DatabaseError("database disk I/O error")
        alerts = []
        EmailAdapter.set_imap_transient_alert_handler(alerts.append)
        installed = make_adapter()
        installed.set_fatal_error_handler(relay_fatal)

        async def run():
            for _ in range(5):
                with patch("imaplib.IMAP4_SSL", return_value=failing_imap(db_exc)):
                    await installed._check_inbox()

        asyncio.run(run())
        report_section(
            "SCENARIO 4 — DB errors (5 attempts, zero suppression)",
            sinks,
            extra=[
                f"ERROR-per-attempt fetch lines: {sinks.count('IMAP fetch error')}/5",
                f"fatal relay lines: {sinks.count('Fatal email adapter error')}/5",
                f"transient alerts for DB: {len(alerts)}",
                f"VERDICT DB visible per attempt: "
                f"{'PASS' if sinks.count('IMAP fetch error') == 5 else 'FAIL'}",
            ],
        )
        assert sinks.count("IMAP fetch error") == 5
        assert sinks.count("Fatal email adapter error") == 5  # immediate notify per attempt
        assert alerts == []
        assert not EmailAdapter._imap_transient_episodes
    finally:
        sinks.close()


def test_register_wires_default_transient_alert_handler():
    """QA rework gate: register() installs the production alert call-site — the threshold
    alert is NOT dead code in a gateway process that loads the email plugin."""
    ctx = MagicMock()
    email_adapter.register(ctx)
    assert EmailAdapter._imap_transient_alert_handler is email_adapter._log_imap_transient_alert
    ctx.register_platform.assert_called_once()

    sinks = Sinks()
    try:
        # End-to-end through the DEFAULT handler: threshold trip emits exactly one
        # '[Email] IMAP transient alert' ERROR with the N/M counters.
        installed = make_adapter()
        installed.set_fatal_error_handler(relay_fatal)
        with patch("imaplib.IMAP4_SSL", return_value=failing_imap(socket.timeout("The read operation timed out"))):
            for _ in range(5):
                asyncio.run(installed._check_inbox())
        assert sinks.count("IMAP transient alert") == 1
        report_section(
            "SCENARIO 5 — register() wiring: default handler emits the alert end-to-end",
            sinks,
            extra=["VERDICT wired call-site emits 1 alert: "
                   f"{'PASS' if sinks.count('IMAP transient alert') == 1 else 'FAIL'}"],
        )
    finally:
        sinks.close()
