"""Plugin adapter failures must never restart the gateway.

A plugin adapter (registry-registered, e.g. email) raising a fatal error is
logged as non-fatal, recorded on the reconnect queue with a bumped
consecutive-failure counter, and the gateway keeps running. Builtin behavior
(exit-with-failure on stranded / last-platform-lost) is unchanged.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.run_adapters import _PLUGIN_CIRCUIT_OPEN_THRESHOLD


class _PluginStubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.EMAIL)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture
def plugin_entry():
    entry = PlatformEntry(
        name="email", label="Email",
        adapter_factory=lambda config: _PluginStubAdapter(),
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    try:
        yield entry
    finally:
        platform_registry.unregister("email")


def _make_runner(config=None):
    runner = object.__new__(GatewayRunner)
    runner.config = config or GatewayConfig(
        platforms={Platform.EMAIL: PlatformConfig(enabled=True, token="test")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner.session_store = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_plugin_fatal_queues_without_restart(plugin_entry):
    """Retryable plugin fatal → reconnect queue, circuit counter bumped, alive."""
    runner = _make_runner()
    runner.stop = AsyncMock()
    adapter = _PluginStubAdapter()
    adapter._set_fatal_error("email_imap_fetch_failed", "IMAP read timeout", retryable=True)
    runner.adapters[Platform.EMAIL] = adapter

    await runner._handle_adapter_fatal_error(adapter)

    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert Platform.EMAIL in runner._failed_platforms
    assert runner._failed_platforms[Platform.EMAIL]["consecutive_errors"] == 1
    assert runner._plugin_circuit_open(Platform.EMAIL) is False


@pytest.mark.asyncio
async def test_plugin_stranded_failure_never_restarts(plugin_entry):
    """Stranded plugin fatal (unqueueable) → non-fatal log, no restart."""
    runner = _make_runner(GatewayConfig(platforms={}))  # queueing impossible
    runner.stop = AsyncMock()
    adapter = _PluginStubAdapter()
    adapter._set_fatal_error("email_imap_fetch_failed", "IMAP read timeout", retryable=True)
    runner.adapters[Platform.EMAIL] = adapter

    await runner._handle_adapter_fatal_error(adapter)

    assert runner._exit_with_failure is False
    runner.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_circuit_opens_after_threshold(plugin_entry):
    """Repeated plugin fatals open the circuit; success clears it."""
    runner = _make_runner()
    runner.stop = AsyncMock()
    for _ in range(_PLUGIN_CIRCUIT_OPEN_THRESHOLD):
        adapter = _PluginStubAdapter()
        adapter._set_fatal_error("email_imap_fetch_failed", "IMAP read timeout", retryable=True)
        runner.adapters[Platform.EMAIL] = adapter
        await runner._handle_adapter_fatal_error(adapter)
        # Simulate the watcher dropping + re-queueing between attempts.
        runner.adapters.pop(Platform.EMAIL, None)

    assert runner._plugin_circuit_open(Platform.EMAIL) is True
    assert runner._exit_with_failure is False
    runner.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_builtin_stranded_failure_still_restarts():
    """Unregistered (builtin) platforms keep the exit-with-failure contract."""
    runner = _make_runner(GatewayConfig(platforms={}))
    runner.config = GatewayConfig(platforms={})

    async def _stop():
        runner._shutdown_event.set()

    runner.stop = AsyncMock(side_effect=_stop)
    adapter = _PluginStubAdapter.__new__(_PluginStubAdapter)
    BasePlatformAdapter.__init__(
        adapter, PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM,
    )
    adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._handle_adapter_fatal_error(adapter)

    assert runner._exit_with_failure is True
    assert runner.stop.await_count == 1
