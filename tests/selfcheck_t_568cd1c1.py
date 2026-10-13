#!/usr/bin/env python3
"""Minimal self-check for t_568cd1c1 (reasoning timeout floor + unpinned terminal cron guard).

The workspace's ``tests/cron/`` collection is broken by a pre-existing gap in
``hermes_cli.config.DEFAULT_CONFIG`` (``gateway.restart`` unconditionally reads
``DEFAULT_CONFIG['gateway']['signal_interrupt_grace_timeout']`` — missing in this
fork, see ``gateway/restart.py:22``).  That gap is unrelated to this task and blocks
every ``tests/cron/`` module at import time, so this script exercises the two pieces
of logic under test directly, without importing ``gateway``.

Run:  python3 tests/selfcheck_t_568cd1c1.py
Exit 0 = all assertions pass.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.reasoning_timeouts import (  # noqa: E402
    get_reasoning_stale_timeout_floor,
    get_reasoning_timeout,
)
from cron.scheduler import (  # noqa: E402
    DRIFT_SKIP_MARKER,
    DRIFT_SKIP_PREFIX,
    guard_unpinned_terminal_toolset,
    unpinned_terminal_toolset_reason,
)


def _base_job(**overrides):
    job = {
        "id": "selfcheck",
        "name": "selfcheck",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def check(label, cond):
    if not cond:
        raise AssertionError(f"FAIL: {label}")
    print(f"ok   {label}")


def main():
    # ── reasoning timeout floor ──────────────────────────────────────────────
    check(
        "floor: get_reasoning_timeout applies the kilo-auto 600s floor over a sub-floor 300s",
        get_reasoning_timeout("custom", "custom/kilo-auto/free", 300, None) == 600,
    )
    check(
        "floor: get_reasoning_timeout applies the kilo-auto 600s floor over a sub-floor 300s",
        get_reasoning_timeout("custom", "custom/kilo-auto/pro", 300, None) == 600,
    )
    check(
        "floor: any other allowlisted reasoning model with explicit 300s -> 1800s",
        get_reasoning_timeout("openai", "openai/o1", 300, None) == 1800,
    )
    check(
        "floor: plain model keeps its configured value verbatim",
        get_reasoning_timeout("openai", "openai/gpt-5", 300, None) == 300,
    )
    check(
        "floor: value at/above the floor is never lowered (kilo-auto 900s)",
        get_reasoning_timeout("custom", "custom/kilo-auto/free", 900, None) == 900,
    )
    check(
        "floor: value at/above the floor is never lowered (o1 3600s)",
        get_reasoning_timeout("openai", "openai/o1", 3600, None) == 3600,
    )
    check(
        "floor: explicit 0 disables the timeout and the floor never re-enables it",
        get_reasoning_timeout("custom", "custom/kilo-auto/free", 0, None) == 0,
    )
    check(
        "floor: None falls back to the default chain (kilo-auto -> 1800s default)",
        get_reasoning_timeout("custom", "custom/kilo-auto/free", None, None) == 1800,
    )
    check(
        "floor: None falls back to the default chain (o1 -> 1800s default)",
        get_reasoning_timeout("openai", "openai/o1", None, None) == 1800,
    )
    check(
        "floor: stale-floor resolver still returns 600 for a known reasoning model (o1)",
        get_reasoning_stale_timeout_floor("openai/o1") == 600,
    )
    check(
        "floor: stale-floor resolver returns None for non-reasoning models",
        get_reasoning_stale_timeout_floor("gpt-5") is None,
    )

    # ── unpinned terminal cron guard ────────────────────────────────────────
    check(
        "guard: terminal + no pins is refused",
        unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["terminal"])) is not None,
    )
    check(
        "guard: terminal + model pin is allowed",
        unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], model="pinned-model")
        ) is None,
    )
    check(
        "guard: terminal + provider pin is allowed",
        unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], provider="pinned-provider")
        ) is None,
    )
    check(
        "guard: blank pins do not count",
        unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], model="   ", provider="")
        ) is not None,
    )
    check(
        "guard: non-terminal job is untouched",
        unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["web", "memory"])) is None,
    )
    check(
        "guard: missing/malformed toolsets are untouched",
        unpinned_terminal_toolset_reason(_base_job()) is None,
    )
    check(
        "guard: toolset name match is case-insensitive",
        unpinned_terminal_toolset_reason(_base_job(enabled_toolsets=["Terminal"])) is not None,
    )
    check(
        "guard: no_agent jobs are exempt",
        unpinned_terminal_toolset_reason(
            _base_job(enabled_toolsets=["terminal"], no_agent=True)
        ) is None,
    )

    # The guard raises a RuntimeError whose message starts with the drift_skip prefix
    # and carries the WARNING marker so the drift-skip delivery branch suppresses the ping.
    try:
        guard_unpinned_terminal_toolset(
            _base_job(enabled_toolsets=["terminal"]), "selfcheck", "selfcheck job"
        )
        raise AssertionError("FAIL: guard should have raised RuntimeError")
    except RuntimeError as exc:
        msg = str(exc)
        check(
            "guard: raises RuntimeError starting with drift_skip prefix",
            msg.startswith(DRIFT_SKIP_PREFIX),
        )
        check(
            "guard: error carries DRIFT_SKIP_MARKER (WARNING)",
            DRIFT_SKIP_MARKER in msg,
        )
        check(
            "guard: error mentions the terminal toolset and the pin requirement",
            "terminal" in msg and "model" in msg and "provider" in msg,
        )

    # An allowed job does not raise.
    try:
        guard_unpinned_terminal_toolset(
            _base_job(enabled_toolsets=["terminal"], model="pinned"), "selfcheck", "selfcheck job"
        )
        check("guard: allowed job does not raise", True)
    except RuntimeError as exc:
        raise AssertionError(f"FAIL: allowed job raised: {exc}") from exc

    print("\nAll self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())