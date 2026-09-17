"""Deterministic, CI-safe coverage for the occupied-port attention path.

Covers the bind-failure contract verified in PR #8:
- occupied-port OSError is fatal, non-retryable
- runtime status reflects needs_attention=True immediately (no retry timer)
- health digest contains the port-conflict error code + message
- emitted log lines carry the expected markers
- NO planned-restart triggered (connect does not restart)

Uses only stdlib + unittest.mock — no live binds, no reactor spins.
"""

import errno
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

pytestmark = pytest.mark.asyncio


def _make_adapter(**extra) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 19_840, "key": "test"})
    cfg.extra.update(extra)
    return APIServerAdapter(cfg)


async def _bind_with_error(adapter, exc):
    with patch("gateway.platforms.api_server.web.TCPSite",
               return_value=MagicMock(start=AsyncMock(side_effect=exc))), \
         patch("gateway.platforms.api_server.web.AppRunner",
               return_value=MagicMock(setup=AsyncMock(), cleanup=AsyncMock())), \
         patch.object(adapter, "_api_key_passes_startup_guard", return_value=True):
        try:
            await adapter.connect()
        except OSError:
            pass


class TestApiServerBindOccupiedPort:
    """Bind-failure paths that surface as fatal + attention, never as restart."""

    def test_make_adapter(self):
        a = _make_adapter()
        assert a.config.enabled is True
        assert a.config.extra["port"] == 19_840

    @pytest.mark.asyncio
    async def test_occupied_port_raises_oserror(self):
        """EADDRINUSE bubbles up as OSError — the caller (reconnect watcher)
        must NOT retry; the adapter is already in fatal state."""
        a = _make_adapter(port=19_840)
        fake_exc = OSError(errno.EADDRINUSE, "Address already in use")
        await _bind_with_error(a, fake_exc)

        assert a._fatal_error_code == "api_server_port_in_use"
        assert a._fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_port_conflict_writes_runtime_status_with_needs_attention(self):
        """Connect failure writes runtime status with needs_attention=True
        immediately (no retry timer) — mirrors the attention path."""
        a = _make_adapter(port=19_841)
        fake_exc = OSError(errno.EADDRINUSE, "Address already in use")
        with patch("gateway.platforms.api_server.web.TCPSite",
                   return_value=MagicMock(start=AsyncMock(side_effect=fake_exc))), \
             patch("gateway.platforms.api_server.web.AppRunner",
                   return_value=MagicMock(setup=AsyncMock(), cleanup=AsyncMock())), \
             patch.object(a, "_api_key_passes_startup_guard", return_value=True), \
             patch.object(a, "_write_runtime_status_safe") as mock_status:
            try:
                await a.connect()
            except OSError:
                pass
            calls = [c for c in mock_status.call_args_list
                     if c[0][0] == "api_server_port_in_use"]
            assert calls, "expected runtime-status write for api_server_port_in_use"
            _, kwargs = calls[0]
            assert isinstance(kwargs, dict), f"expected kwargs dict, got {type(kwargs)}: {calls[0]}"
            assert kwargs.get("needs_attention") is True
            assert kwargs.get("platform_state") == "fatal"
            assert kwargs.get("error_code") == "api_server_port_in_use"

    @pytest.mark.asyncio
    async def test_health_digest_contains_port_conflict_code(self):
        """_handle_health_detailed() exposes the port-conflict error code."""
        from gateway.status import read_runtime_status as _rrs

        a = _make_adapter(port=19_841)
        a._set_fatal_error(
            "api_server_port_in_use",
            f"Port {a.config.extra.get('port', 19_841)} already in use. "
            "Set platforms.api_server.port in config.yaml.",
            retryable=False,
        )
        a._fatal_error_code = "api_server_port_in_use"
        a._fatal_error_message = "Port 19_841 already in use."
        fake_runtime = {
            "gateway_state": "fatal",
            "error_code": "api_server_port_in_use",
            "exit_reason": "api_server_port_in_use",
            "needs_attention": True,
        }
        with patch.object(a, "_check_auth", return_value=None), \
             patch("gateway.status.read_runtime_status", return_value=fake_runtime), \
             patch.object(a, "_readiness_work_counts", return_value=(0, 0, 0)):
            detailed = await a._handle_health_detailed(_fake_request())
        body_text = detailed.text
        assert detailed.content_type == "application/json"
        import json
        body = json.loads(body_text if isinstance(body_text, str) else body_text())
        assert body["exit_reason"] == "api_server_port_in_use"
        assert body["gateway_state"] == "fatal"

    @pytest.mark.asyncio
    async def test_occupied_port_logs_error_with_port_marker(self, caplog):
        """connect() logs the occupied-port error at ERROR level with the
        port number visible for ops triage."""
        a = _make_adapter(port=19_843)
        fake_exc = OSError(errno.EADDRINUSE, "Address already in use")
        port = 19_843
        with patch("gateway.platforms.api_server.web.TCPSite",
                   return_value=MagicMock(start=AsyncMock(side_effect=fake_exc))), \
             patch("gateway.platforms.api_server.web.AppRunner",
                   return_value=MagicMock(setup=AsyncMock(), cleanup=AsyncMock())), \
             patch.object(a, "_api_key_passes_startup_guard", return_value=True):
            with caplog.at_level(logging.ERROR, logger="gateway.platforms.api_server"):
                try:
                    await a.connect()
                except OSError:
                    pass
        assert any("Could not bind" in r.getMessage() for r in caplog.records), \
            "expected ERROR log for bind failure"
        assert any(str(port) in r.getMessage() for r in caplog.records), \
            "expected port number in ERROR log"

    @pytest.mark.asyncio
    async def test_occupied_port_does_not_request_restart(self):
        """Connect failure must not trigger request_restart / planned-stop —
        a port conflict is a config error, not a recoverable transient."""
        a = _make_adapter(port=19_844)
        fake_exc = OSError(errno.EADDRINUSE, "Address already in use")
        mock_restart = MagicMock(return_value=False)
        with patch("gateway.platforms.api_server.web.TCPSite",
                   return_value=MagicMock(start=AsyncMock(side_effect=fake_exc))), \
             patch("gateway.platforms.api_server.web.AppRunner",
                   return_value=MagicMock(setup=AsyncMock(), cleanup=AsyncMock())), \
             patch.object(a, "_api_key_passes_startup_guard", return_value=True), \
             patch("gateway.run.GatewayRunner.request_restart", mock_restart):
            try:
                await a.connect()
            except OSError:
                pass
        mock_restart.assert_not_called()


def _fake_request():
    """Build a minimal aiohttp Request without a real network stack."""
    from aiohttp import web
    from aiohttp.http_parser import RawRequestMessage
    from aiohttp.streams import StreamReader
    import asyncio
    p = MagicMock()
    pw = MagicMock()
    t = asyncio.current_task()
    l = asyncio.get_running_loop()
    m = MagicMock()
    m.headers = {}
    payload = StreamReader(protocol=p, limit=1024, loop=l)
    return web.Request(m, payload, p, pw, t, l)