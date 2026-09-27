"""The test suite must never write into the operator's real Hermes logs.

`hermes_cli/main.py` calls `setup_logging()` at module scope, which resolves
`get_hermes_home()` and attaches rotating file handlers to the ROOT logger.
Importing it - which many test modules do, directly or transitively - wires
the whole pytest session's logging to `<HERMES_HOME>/logs/agent.log`.

If HERMES_HOME is not already sandboxed at that moment, that is the
operator's real log. Measured on a live install, 126 warnings in a personal
`agent.log` came from test runs rather than the running gateway: phantom
`FakeTree` Discord failures and `rejected invalid API key` entries from
`test_api_server_runs.py`. Noise like that makes genuine warnings hard to
find precisely when someone is debugging.

The per-test env fixture cannot close this: fixtures run after collection has
imported the test modules, and by then the handler holds an absolute path.
`tests/conftest.py` sets HERMES_HOME at module scope for that reason - this
guards the property so a refactor cannot quietly undo it.

A second shape is worse: a home the production check used to read as "custom"
(`HERMES_HOME=/opt/data` with `HOME=/opt/data`, or
`HERMES_HOME=/opt/data/profiles/<name>` with `HOME=<profile>/home`) skipped the
sandbox entirely, so collection anchored the handlers at the LIVE
`<install>/logs`. Measured 2026-09-25: eight `[api_server] Refusing to start`
ERROR records in the live `errors.log`, every one of them emitted by
`tests/gateway/test_api_server.py` exercising the fails-closed key guard.
`TestLauncherShapedHomeIsSandboxed` and `TestLiveLogHandlerGuard` below pin the
two halves of the fix.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _real_hermes_home() -> Path:
    """Where the operator's logs live, ignoring any test sandboxing."""
    return Path.home() / ".hermes"


def _all_file_destinations() -> list[str]:
    """Every file path the root logger can reach, including via a QueueHandler.

    Logging is routed through a queue, so the file handlers hang off the
    listener rather than the root logger - checking `root.handlers` alone
    reports nothing and looks falsely clean.
    """
    seen: list[str] = []

    def collect(handlers) -> None:
        for handler in handlers or ():
            path = getattr(handler, "baseFilename", None)
            if path:
                seen.append(str(path))
            listener = getattr(handler, "listener", None)
            if listener is not None:
                collect(getattr(listener, "handlers", ()))

    collect(logging.getLogger().handlers)

    try:
        import hermes_logging

        listener = getattr(hermes_logging, "_queue_listener", None)
        if listener is not None:
            collect(getattr(listener, "handlers", ()))
    except Exception:
        pass

    return seen


class TestLogIsolation:
    def test_hermes_home_is_sandboxed_before_imports(self):
        # Deliberately NOT os.environ: by test time the per-test `_isolate_env`
        # fixture has sandboxed HERMES_HOME, so reading it here would pass even
        # with the conftest block deleted. Assert the value captured at conftest
        # import, which is the moment that actually matters.
        from tests.conftest import HERMES_HOME_AT_CONFTEST_IMPORT as home

        assert home, "conftest must set HERMES_HOME before test modules import"
        assert Path(home).resolve() != _real_hermes_home().resolve(), (
            f"HERMES_HOME pointed at the operator's real home ({home}) when "
            "conftest loaded; import-time setup_logging() writes to their agent.log"
        )

    def test_importing_the_cli_does_not_target_the_real_logs(self):
        pytest.importorskip("hermes_cli.main")

        real_logs = str(_real_hermes_home() / "logs")
        offenders = [p for p in _all_file_destinations() if p.startswith(real_logs)]

        assert offenders == [], (
            "the test session is writing into the operator's real Hermes logs:\n  "
            + "\n  ".join(offenders)
        )


class TestLauncherShapedHomeIsSandboxed:
    """A supervisor-shaped HERMES_HOME is production, not a custom override.

    `HERMES_HOME=/opt/data` with `HOME=/opt/data` (Docker root, default-profile
    worker) and `HERMES_HOME=<root>/profiles/<name>` with `HOME=<profile>/home`
    are how every deployment hands its own install to a child shell. Reading
    them as "custom" disabled the session sandbox, which is how the live
    `errors.log` collected pytest ERROR records on 2026-09-25.
    """

    @staticmethod
    def _fake_platform_root(monkeypatch, root: Path) -> None:
        """Point the "real platform root" at a temp path (no live probing)."""
        import hermes_state_guard

        monkeypatch.setattr(hermes_state_guard, "_real_platform_state_root", lambda: root)

    def test_docker_root_with_home_inside_it_is_production(self, tmp_path, monkeypatch):
        from tests import conftest

        install = tmp_path / "data"
        install.mkdir()
        # HOME == HERMES_HOME: `~/.hermes` resolves inside the install root.
        self._fake_platform_root(monkeypatch, install / ".hermes")

        assert conftest._hermes_home_points_at_production(str(install)) is True

    def test_profile_dir_with_profile_home_is_production(self, tmp_path, monkeypatch):
        from tests import conftest

        profile = tmp_path / "data" / "profiles" / "joe"
        profile.mkdir(parents=True)
        # HOME = <profile>/home → the platform root lives inside the profile dir.
        self._fake_platform_root(monkeypatch, profile / "home" / ".hermes")

        assert conftest._hermes_home_points_at_production(str(profile)) is True

    def test_override_outside_the_platform_root_is_honoured(self, tmp_path, monkeypatch):
        """A throwaway home must keep working — some suites set one on purpose."""
        from tests import conftest

        self._fake_platform_root(monkeypatch, tmp_path / "home" / ".hermes")
        custom = tmp_path / "scratch-home"
        custom.mkdir()

        assert conftest._hermes_home_points_at_production(str(custom)) is False


class TestLiveLogHandlerGuard:
    """Handlers bound before the sandbox must be detached, not left writing."""

    @staticmethod
    def _live_dir(tmp_path: Path) -> Path:
        live = tmp_path / "install" / "logs"
        live.mkdir(parents=True)
        return live.resolve()

    def test_handler_pointed_at_a_live_log_dir_is_detached(self, tmp_path, monkeypatch):
        from tests import conftest

        live = self._live_dir(tmp_path)
        handler = logging.FileHandler(live / "errors.log")
        root = logging.getLogger()
        root.addHandler(handler)
        monkeypatch.setattr(conftest, "_live_hermes_log_dirs", lambda: [live])
        try:
            detached = conftest._detach_live_log_handlers()

            assert detached == [str(live / "errors.log")]
            assert handler not in root.handlers
        finally:
            if handler in root.handlers:
                root.removeHandler(handler)

    def test_handler_outside_the_live_log_dir_is_kept(self, tmp_path, monkeypatch):
        """The session's own log (sandbox tempdir) must survive the guard."""
        from tests import conftest

        live = self._live_dir(tmp_path)
        session_dir = tmp_path / "hermes-test-home-session"
        session_dir.mkdir()
        handler = logging.FileHandler(session_dir / "agent.log")
        root = logging.getLogger()
        root.addHandler(handler)
        monkeypatch.setattr(conftest, "_live_hermes_log_dirs", lambda: [live])
        try:
            assert conftest._detach_live_log_handlers() == []
            assert handler in root.handlers
        finally:
            if handler in root.handlers:
                root.removeHandler(handler)
            handler.close()


# The launcher configures logging in its own process and then hands that process
# (or its install root) to whatever runs next. `sitecustomize` is imported by the
# interpreter at startup, before pytest exists — exactly the launcher's ordering.
# (A `-p` plugin cannot be used here: importing hermes_cli.main parses sys.argv
# and treats `-p NAME` as a profile selection.)
_PROBE_SITECUSTOMIZE = '''\
import hermes_cli.main  # noqa: F401 — module-scope setup_logging() wires file handlers
'''

# Runs inside the launcher-shaped session. Emits the guard record that showed up
# in the live errors.log, then reports what the conftest guard did.
_PROBE_TEST = '''\
import json
import os


def test_probe():
    from gateway.platforms.api_server import APIServerAdapter

    class _Stub:
        name = "api_server"
        _host = "127.0.0.1"

        def __init__(self, key):
            self._api_key = key

    # The exact refusal that leaked into the operator's errors.log.
    assert APIServerAdapter._api_key_passes_startup_guard(_Stub("")) is False

    import tests.conftest as conftest

    print(
        "PROBE_SANDBOX="
        + str(getattr(conftest, "HERMES_HOME_AT_CONFTEST_IMPORT", ""))
    )
    print("PROBE_KANBAN_ROOT=" + str(getattr(conftest, "_REAL_KANBAN_ROOT", "")))
    print(
        "PROBE_DETACHED="
        + json.dumps(getattr(conftest, "DETACHED_LIVE_LOG_HANDLERS", []))
    )
'''


class TestLauncherSessionLeavesTheInstallLogAlone:
    """End-to-end: a launcher-shaped session writes nothing into its install."""

    def _run_launcher_session(self, tmp_path):
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        (install / "logs").mkdir()
        (install / "config.yaml").write_text("# fake launcher install\n")
        (tmp_path / "sitecustomize.py").write_text(_PROBE_SITECUSTOMIZE)
        probe = tmp_path / "test_probe_launcher_session.py"
        probe.write_text(_PROBE_TEST)

        env = dict(os.environ)
        for key in list(env):
            if key.startswith("HERMES_SESSION") or key.startswith("HERMES_TEST"):
                env.pop(key)
        env["HERMES_HOME"] = str(install)
        env["HOME"] = str(install / "home")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(tmp_path), str(_PROJECT_ROOT), env.get("PYTHONPATH", "")]
        )
        # tests/conftest.py as a plugin without touching argv (see above).
        env["PYTEST_PLUGINS"] = "tests.conftest"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", str(probe)],
            cwd=str(_PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return install, proc

    @staticmethod
    def _probe_value(out: str, key: str):
        for line in out.splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
        return None

    def test_install_log_gets_no_test_records(self, tmp_path):
        install, proc = self._run_launcher_session(tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out[-3000:]

        # The sandbox engaged at conftest import: HERMES_HOME was redirected
        # before any test module could anchor a handler at the live install.
        sandbox_home = self._probe_value(out, "PROBE_SANDBOX")
        assert sandbox_home, out[-3000:]
        assert sandbox_home != str(install), (
            "the launcher-shaped HERMES_HOME was left in place; the session wrote "
            "into a live install"
        )
        assert "hermes-test-home-" in sandbox_home, sandbox_home

        # The live kanban root is still captured, so the write guard keeps
        # denying the operator's board for this deployment shape too.
        kanban_root = self._probe_value(out, "PROBE_KANBAN_ROOT")
        assert kanban_root == str(install), (
            "the kanban deny-list lost the install root for a launcher-shaped "
            f"home — the guard would fail open (got {kanban_root!r})"
        )

        # ...and the handlers the launcher bound were detached rather than left
        # writing through an already-open file.
        detached = json.loads(self._probe_value(out, "PROBE_DETACHED") or "[]")
        assert set(detached) == {
            str(install / "logs" / "agent.log"),
            str(install / "logs" / "errors.log"),
        }, f"detached={detached}\n{out[-3000:]}"

        leaked = [
            f"{path.name}: {line}"
            for path in sorted((install / "logs").glob("*.log*"))
            for line in path.read_text(errors="replace").splitlines()
            if "Refusing to start" in line
        ]
        assert leaked == [], "records leaked into the install log:\n  " + "\n  ".join(leaked)
