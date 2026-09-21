"""Regression tests for transient IMAP timeout episodes."""

import asyncio
import imaplib
import logging
import os
import socket
import sqlite3
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.email import adapter as email_adapter
from plugins.platforms.email.adapter import EmailAdapter


@pytest.fixture
def adapter():
    EmailAdapter._imap_transient_episodes.clear()
    EmailAdapter.set_imap_transient_alert_handler(None)
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=True):
        instance = EmailAdapter(email_adapter.PlatformConfig(enabled=True))
        instance._write_runtime_status_safe = MagicMock()
        yield instance
    EmailAdapter._imap_transient_episodes.clear()
    EmailAdapter.set_imap_transient_alert_handler(None)


@pytest.mark.parametrize("exc,phase,expected", [
    (socket.timeout("The read operation timed out"), "fetch", True),
    (TimeoutError("handshake operation timed out"), "connect", True),
    (ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"), "login", True),
    (BrokenPipeError("write failed: errno 104"), "search", True),
    (ConnectionResetError("errno 10054"), "select", True),
    (ssl.SSLError("CERTIFICATE_VERIFY_FAILED"), "connect", False),
    (imaplib.IMAP4.error("[AUTHENTICATIONFAILED] invalid credentials"), "login", False),
    (sqlite3.DatabaseError("database disk I/O error"), "fetch", False),
    (ValueError("invalid IMAP response"), "fetch", False),
    (TimeoutError("read operation timed out"), "parse", False),
])
def test_imap_transient_classification_is_strict(adapter, exc, phase, expected):
    assert EmailAdapter._is_imap_transient_error(exc, phase) is expected


def test_repeated_transient_errors_emit_one_fatal_and_suppress_error_logs(adapter, caplog):
    notified = AsyncMock()
    adapter.set_fatal_error_handler(notified)
    mock_imap = MagicMock()
    mock_imap.login.side_effect = TimeoutError("The read operation timed out")

    with patch("imaplib.IMAP4_SSL", return_value=mock_imap), caplog.at_level(logging.DEBUG):
        for _ in range(3):
            asyncio.run(adapter._check_inbox())

    fetch_records = [r for r in caplog.records if "IMAP fetch error" in r.getMessage()]
    assert [r.levelno for r in fetch_records] == [
        logging.ERROR, logging.DEBUG, logging.DEBUG
    ]
    assert "x2, same class" in fetch_records[1].getMessage()
    notified.assert_awaited_once()
    assert adapter.fatal_error_code == "email_imap_transient_error"


def test_alert_is_emitted_once_per_episode_and_rearmed_after_recovery(adapter):
    alerts = []
    EmailAdapter.set_imap_transient_alert_handler(alerts.append)

    for _ in range(5):
        adapter._record_imap_transient_error(
            "fetch", TimeoutError("The read operation timed out")
        )
    for _ in range(3):
        adapter._record_imap_transient_error(
            "fetch", TimeoutError("The read operation timed out")
        )

    assert len(alerts) == 1
    assert alerts[0]["count"] == 5
    assert alerts[0]["window_count"] == 5

    adapter._end_imap_transient_episode()
    for _ in range(5):
        adapter._record_imap_transient_error(
            "fetch", TimeoutError("The read operation timed out")
        )

    assert len(alerts) == 2
    assert alerts[1]["count"] == 5


def test_recovery_ends_episode_with_one_summary(adapter, caplog):
    adapter._record_imap_transient_error(
        "fetch", TimeoutError("The read operation timed out")
    )
    adapter._record_imap_transient_error(
        "fetch", TimeoutError("The read operation timed out")
    )

    with caplog.at_level(logging.INFO):
        adapter._end_imap_transient_episode()

    assert any(
        "IMAP fetch recovered after 2 consecutive TimeoutError error(s)" in r.getMessage()
        for r in caplog.records
    )
    assert not EmailAdapter._imap_transient_episodes


def test_non_transient_auth_failures_remain_fully_visible(adapter, caplog):
    notified = AsyncMock()
    adapter.set_fatal_error_handler(notified)
    mock_imap = MagicMock()
    mock_imap.login.side_effect = imaplib.IMAP4.error(
        "[AUTHENTICATIONFAILED] invalid credentials"
    )

    with patch("imaplib.IMAP4_SSL", return_value=mock_imap), caplog.at_level(logging.ERROR):
        for _ in range(2):
            asyncio.run(adapter._check_inbox())

    assert notified.await_count == 2
    fetch_records = [r for r in caplog.records if "IMAP fetch error" in r.getMessage()]
    assert [r.levelno for r in fetch_records] == [logging.ERROR, logging.ERROR]
    assert not EmailAdapter._imap_transient_episodes


def test_successful_imap_command_ends_transient_episode(adapter, caplog):
    adapter._record_imap_transient_error(
        "fetch", TimeoutError("The read operation timed out")
    )
    mock_imap = MagicMock()
    mock_imap.uid.return_value = ("OK", [b""])

    with patch("imaplib.IMAP4_SSL", return_value=mock_imap), caplog.at_level(logging.INFO):
        assert adapter._fetch_new_messages() == []

    assert any("IMAP fetch recovered after 1 consecutive" in r.getMessage()
               for r in caplog.records)
    assert not EmailAdapter._imap_transient_episodes
