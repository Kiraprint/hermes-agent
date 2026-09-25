"""Per-model stale-timeout FLOOR for known reasoning models.

Reasoning models routinely exceed the default chat-model stale detectors (stream 180s,
non-stream 90s): upstream proxies idle-kill the stream mid-think, surfacing as
``BrokenPipeError``/``RemoteProtocolError``. The stale-detector scaling applies
``max(default, floor)`` from :func:`get_reasoning_stale_timeout_floor`, so this never
overrides explicit per-model ``stale_timeout_seconds``/``request_timeout_seconds`` (that
branch never calls it), never lowers a threshold, and is ``None`` for non-allowlisted models.
"""

from __future__ import annotations

import re
from typing import Optional


# floor_seconds -> slugs. Order irrelevant — longest slug wins at match time.
_REASONING_STALE_TIMEOUT_FLOORS: dict[int, tuple[str, ...]] = {
    600: (
        # NVIDIA Nemotron behind hosted NIM: documented 60-180s upstream idle kill.
        "nemotron-3-ultra", "nemotron-3-super",
        # DeepSeek R1 / V4 (reasoning_content streamed before final content).
        # ``deepseek-flash`` is the version-less canonical Flash id (2026-09 Flash refresh);
        # ``deepseek-v4-flash`` still aliases onto it server-side.
        "deepseek-r1", "deepseek-reasoner", "deepseek-flash", "deepseek-v4-flash", "deepseek-v4.1-flash", "deepseek-v4-pro",
        # OpenAI o-series: each variant enumerated so bare ``o1`` cannot over-match ``olmo-1``.
        "o1", "o1-mini", "o1-pro", "o1-preview", "o3", "o3-pro",
        # Mythos-class named models (claude-fable-5): 1M ctx + 128K output, a heavier thinking
        # phase than the numbered line — otherwise the stale detector trips the circuit breaker.
        "claude-fable",
    ),
    300: (
        "nemotron-3-nano", "nemotron-3.5-lightning", "qwq-32b", "o3-mini", "o4-mini",
        # xAI Grok: explicit reasoning pairs only, so bare ``grok-3``/``grok-4`` fast variants
        # don't inherit the floor.
        "grok-4-fast-reasoning", "grok-4.20-reasoning", "grok-4.5", "grok-4.6",
        # "Ox Alpha" stealth reasoning model (OpenRouter / OpenCode Zen slugs); Thinking
        # Machines Inkling (covers inkling-small and :free SKUs).
        "ox-alpha", "x-preview-f-free", "inkling",
        # Qwen3.8 generation — reasoning Qwen3.8 models (qwen3.8-27b, qwen3.8-27b-nvfp4)
        # on local vLLM: a non-streaming call can need 3-5+ minutes to first byte
        # (ctlab-doom-run-monitor 180s "no response" timeouts, incident t_3a513c34),
        # past the 180s qwen3 family floor.  Longest-slug match means only
        # qwen3.8* models get the 300s floor; plain qwen3 instruct/thinking variants
        # keep the 180s floor.
        "qwen3.8",
    ),
    # Anthropic Claude 4.x+ thinking variants (anchored so 3.x never matches).
    240: ("claude-opus-4", "claude-opus-5"),
    # qwen3 family: instruct variants also match — a slightly longer wait on a hung provider
    # beats a pattern (``qwen3-.*-thinking``) that breaks on the next naming shape.
    180: ("qwen3", "claude-sonnet-5", "claude-sonnet-4.5", "claude-sonnet-4.6", "grok-4-fast-non-reasoning"),
}


# Pre-compiled once at import (immutable afterwards — safe under free-threaded Python).
# Right anchor: end-of-string or a slug separator; ``:`` because OpenRouter routing suffixes
# (``:free``, ``:nitro``) attach directly to the slug. Longest-first so ``o3-mini`` beats ``o3``.
_SORTED_REASONING_FLOORS: list[tuple[str, float, re.Pattern[str]]] = [
    (slug, floor, re.compile(r"^" + re.escape(slug) + r"(?:$|[\-._:])"))
    for slug, floor in sorted(
        ((slug, floor) for floor, slugs in _REASONING_STALE_TIMEOUT_FLOORS.items() for slug in slugs),
        key=lambda kv: -len(kv[0]),
    )
]


def get_reasoning_stale_timeout_floor(model: object) -> Optional[float]:
    """Stale-timeout floor (seconds) for a known reasoning model, else ``None``.

    The aggregator prefix (up to the last ``/``) is stripped and the slug matched
    start-anchored with an end-or-separator right anchor, so ``qwen3-235b`` matches ``qwen3``
    but ``some-other-qwen3`` and ``llama-4-70b-o1-preview`` do not.
    """
    if not model or not isinstance(model, str):
        return None
    name = model.strip().lower().rsplit("/", 1)[-1]
    for _slug, floor, pattern in _SORTED_REASONING_FLOORS:
        if pattern.search(name):
            return float(floor)
    return None


# ── Unified per-request timeout floor for reasoning models (t_568cd1c1) ─────────
#
# Reasoning / aggregator-routing calls can legitimately need minutes per call; the
# pre-incident 300-second per-model timeout (incident t_ec7a0c1c) cut them off
# mid-think. ``get_reasoning_timeout()`` raises any sub-floor explicit timeout up to
# the floor. The floor is applied to reasoning-family models only, so an operator's
# explicit per-model value for a plain model is never overridden:
#   * Kilo-gateway auto routing (``.../kilo-auto/...``, e.g. ``custom/kilo-auto/free``
#     / ``custom/kilo-auto/pro``) -> 600s
#   * any other allowlisted reasoning model -> 1800s
# Preserved rules: an explicit 0 (disabled timeout) stays 0, and a missing value
# (None) resolves to the provider/model default chain (``default`` below).

#: Kilo-gateway auto-router: its upstream idle-kill window is shorter than a full
#: reasoning think, so its floor is lower than the universal one.
_KILO_AUTO_REQUEST_TIMEOUT_FLOOR_SECONDS = 600.0
#: Universal floor for any other reasoning model: a slow / queueing / thinking call
#: must not be cut off at a sub-floor per-request timeout.
_DEFAULT_REASONING_REQUEST_TIMEOUT_FLOOR_SECONDS = 1800.0


def _is_kilo_auto_router(model: object) -> bool:
    """True when *model* routes through the Kilo ``kilo-auto`` aggregator.

    The Kilo auto slug appears as a ``/``-separated component of the model string
    (``custom/kilo-auto/free``, ``kilo-auto/pro``); any component starting with
    ``kilo-auto`` matches so future SKUs keep matching.
    """
    if not model or not isinstance(model, str):
        return False
    return any(comp.startswith("kilo-auto") for comp in model.strip().lower().split("/"))


def get_reasoning_timeout(
    provider: object, model: object, timeout: Optional[float],
    default: Optional[float] = None,
) -> float:
    """Effective per-request timeout for *provider*/*model*, with the reasoning floor.

    * ``timeout`` <= 0 (explicit off) -> ``0.0``: the floor never re-enables a
      timeout the operator explicitly disabled.
    * ``timeout is None`` (nothing explicitly configured) -> the provider/model
      default: *default* when the caller supplies the resolved default chain, else
      the built-in 1800s; a reasoning model's default is still raised to its floor.
    * ``0 < timeout < floor`` -> the floor (never lowered); ``timeout >= floor``
      -> unchanged.
    Non-reasoning models carry no floor: their values are honored verbatim.
    """
    floor: Optional[float]
    if _is_kilo_auto_router(model):
        floor = _KILO_AUTO_REQUEST_TIMEOUT_FLOOR_SECONDS
    elif get_reasoning_stale_timeout_floor(model) is not None:
        floor = _DEFAULT_REASONING_REQUEST_TIMEOUT_FLOOR_SECONDS
    else:
        floor = None
    if timeout is not None:
        try:
            explicit = float(timeout)
        except (TypeError, ValueError):
            explicit = None
        if explicit is not None:
            if explicit <= 0:
                return 0.0
            if floor is not None:
                return max(explicit, floor)
            return explicit
    base = float(default) if default is not None else _DEFAULT_REASONING_REQUEST_TIMEOUT_FLOOR_SECONDS
    if floor is not None:
        return max(base, floor)
    return base
