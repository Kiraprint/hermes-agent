"""Durable per-``(provider, model)`` rate-limit cooldowns for the fallback walk.

Live evidence (2026-09-28 11:23:51 / 14:58:30, session 20260928_104007_a4d38f0e):

    Fallback activated: anymodel/ds/deepseek-v4-flash → anymodel/ds/deepseek-v4-flash (llm-router)
    Streaming failed before delivery: Error code: 429 - {... 'code': 'model_cooldown' ...}
    API call failed (attempt 1/3) ... model=anymodel/ds/deepseek-v4-flash (llm-router)
    Fallback activated: anymodel/ds/deepseek-v4-flash → bai/mimo-v2.6-flash (llm-router)

The chain's FIRST entry re-selects the very model that just returned 429 (it is the
only ``llm-router`` entry for that model, and ``should_skip_candidate`` compares
provider+base_url identity, not the model slug), so the retry loop spends an extra
round trip on a model the router has already benched — three provider contacts and
two full request bodies per rate-limit incident.

This module records the 429 window for that ``(provider, model)`` pair in a sidecar
file and lets ``_should_skip_fallback_candidate`` skip the pair until it expires,
so the walk lands on the next healthy entry on the first try. The sidecar (not the
database, not in-memory state) is what makes the cooldown survive a process
restart, mirroring the kanban circuit-breaker sidecar pattern (t_eeba7a3d, acf432c30).

Fail-open by construction: every read/write is wrapped, a missing/corrupt file reads
as "no cooldowns armed", and an unwritable file only loses the cooldown — it can
never raise into the retry loop or restart a gateway.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SIDECAR_FILENAME = ".kanban_rate_limit.json"

# Cooldown applied when the provider declares no usable retry-after window.
DEFAULT_COOLDOWN_SECONDS = 600

# Bounds on any cooldown, however it was derived. The lower bound keeps a
# sub-minute provider hint from re-arming the same entry on the very next call;
# the upper bound stops a malformed "retry after 9999999" from benching a model
# for the rest of the day.
MIN_COOLDOWN_SECONDS = 300
MAX_COOLDOWN_SECONDS = 600

# Only the last N entries are kept; a long-lived process walking many models must
# not grow the file without bound. Expired entries are dropped on every write.
_MAX_ENTRIES = 64

# Free-tier throttling. The router reports these as HTTP 429 with either
# ``free_rate_limited`` in the body or a ``type: rate_limit_error`` envelope.
_FREE_TIER_PATTERNS = (
    "free_rate_limited",
    "rate limit exceeded on free",
    "free tier rate limit",
    "too many requests on free",
)

# 429 envelopes shaped like {"error": {"code": "model_cooldown", ...}} — the router's
# own bench of a model whose credentials are cooling down.
_COOLDOWN_CODES = ("model_cooldown", "free_rate_limited", "rate_limit_exceeded", "resource_exhausted")

_RETRY_AFTER_RE = re.compile(r"retry[-_\s]?after[\"'\s:=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_JSON_FIELD_RE = re.compile(
    r"[\"']?(resets?(?:_in(?:_seconds)?|_seconds|_after)?|reset[_\s]?in(?:_seconds)?)"
    r"[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|second|m|min|minute|h|hour)?",
    re.IGNORECASE,
)
_DURATION_TEXT_RE = re.compile(
    r"reset(?:s|ting)?\s+after\s+(?:(\d+)\s*(h|hour)s?\s*)?(?:(\d+)\s*(m|min|minute)s?\s*)?(?:(\d+)\s*(s|sec|second)s?)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cooldown:
    """One armed rate-limit window for a ``(provider, model)`` pair."""

    provider: str
    model: str
    reset_at: float
    retry_after: float

    def remaining(self, now: float) -> float:
        return max(0.0, self.reset_at - now)

    def percent(self, now: float) -> int:
        """Progress through the window, for the skip log line."""
        span = self.retry_after or DEFAULT_COOLDOWN_SECONDS
        return int(min(100.0, max(0.0, 100.0 * self.remaining(now) / span)))


def _sidecar_path() -> Optional[Path]:
    """``HERMES_HOME/.kanban_rate_limit.json``; None when HERMES_HOME is unusable."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / SIDECAR_FILENAME
    except Exception as exc:
        logger.debug("rate-limit sidecar path unavailable: %s", exc)
        return None


def _entry_key(provider: str, model: str, base_url: str = "") -> str:
    """Identity of the *physical* backend that got throttled.

    Provider labels are config aliases, not identity: the live config routes
    ``provider: custom`` and ``provider: llm-router`` at the same
    ``http://127.0.0.1:20130/v1`` router, so keying on the label alone would arm
    the cooldown under ``custom`` and never match the ``llm-router`` chain entry.
    ``base_url`` is therefore part of the key whenever we can see it (mirrors the
    ``BackendIdentity`` rule that provider aliases are not endpoints).
    """
    endpoint = _normalize_base_url(base_url)
    return f"{endpoint}|{(model or '').strip().lower()}" if endpoint else \
        f"{(provider or '').strip().lower()}:{(model or '').strip().lower()}"


def _normalize_base_url(base_url: str) -> str:
    """Lowercase host:port + path, scheme-insensitively, for identity comparison."""
    if not base_url or not isinstance(base_url, str):
        return ""
    cleaned = base_url.strip().lower().rstrip("/")
    cleaned = re.sub(r"^https?://", "", cleaned)
    return cleaned.split("?", 1)[0]


def _read_state() -> Dict[str, Any]:
    """Parsed sidecar contents; ``{}`` for missing, unreadable or malformed files."""
    path = _sidecar_path()
    if path is None:
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.debug("rate-limit sidecar unreadable (%s): %s", path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("rate-limit sidecar %s is malformed (%s); treating it as empty", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: Dict[str, Any]) -> bool:
    """Atomically replace the sidecar. Never raises: a lost cooldown beats a broken turn.

    Returns False when the cooldown could not be persisted (unwritable home, missing
    directory), so the caller can report an unarmed cooldown instead of a phantom one.
    """
    path = _sidecar_path()
    if path is None:
        return False
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("rate-limit sidecar write failed (%s): %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _coerce_seconds(value: Any) -> Optional[float]:
    """Positive float from an int/float/numeric-string; None otherwise."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not (seconds > 0):
        return None
    return 3600.0 if seconds > 86400 else seconds  # an epoch-looking value is a duration bug; cap it


_UNIT_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "m": 60.0, "min": 60.0, "minute": 60.0,
                 "h": 3600.0, "hour": 3600.0}


def parse_retry_after_seconds(text: Any) -> Optional[float]:
    """Parse a Retry-After style hint into raw seconds (no clamping).

    Handles plain numbers, ``retry_after: 90`` / ``resets_in_seconds: 94`` fields,
    multi-unit durations (``reset after 1m 33s`` → 93), and bare ``60s``/``2 min``.
    Returns None when nothing usable is present — the caller then applies the default.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return _coerce_seconds(text)
    if not isinstance(text, str) or not text.strip():
        return None
    blob = text.strip()

    direct = _coerce_seconds(blob)
    if direct is not None:
        return direct

    # Multi-unit durations first, so "1m 33s" is 93 and not just the leading 60.
    match = _DURATION_TEXT_RE.search(blob)
    if match:
        total = sum(float(group or 0) * unit for group, unit in
                    ((match.group(1), 3600.0), (match.group(3), 60.0), (match.group(5), 1.0)))
        if total > 0:
            return total

    # Relative JSON fields ("reset_seconds": 94, retry_after: 300) BEFORE the bare
    # retry-after sweep: an absolute ISO retry_after must never be read as "2026 seconds".
    match = _JSON_FIELD_RE.search(blob)
    if match and _coerce_seconds(match.group(2)) is not None:
        unit = (match.group(3) or "").lower()
        return _coerce_seconds(match.group(2)) * _UNIT_SECONDS.get(unit, 1.0)

    match = _RETRY_AFTER_RE.search(blob)
    if match and _coerce_seconds(match.group(1)) is not None:
        # Skip a timestamp-shaped value (>= 2000 reads as a year, not a duration).
        value = _coerce_seconds(match.group(1))
        if value < 2000:
            return value

    unit_match = re.search(r"(\d+(?:\.\d+)?)\s*(h|hour|m|min|minute|s|sec|second)s?\b", blob, re.IGNORECASE)
    if unit_match:
        return float(unit_match.group(1)) * _UNIT_SECONDS.get(unit_match.group(2).lower(), 1.0)
    return None


def cooldown_seconds_from_error(message: str, error_context: Optional[Dict[str, Any]] = None) -> float:
    """Cooldown to arm for a rate-limited error, clamped to [MIN, MAX].

    Prefers the provider's own window: a relative ``retry_after``/``resets_in_seconds``
    field, then the free-text duration in the message, then an absolute ``reset_at``
    (whose remaining time is all that matters). Falls back to DEFAULT.
    """
    ctx = error_context or {}

    for key in ("retry_after", "resets_in_seconds", "reset_seconds"):
        parsed = parse_retry_after_seconds(ctx.get(key))
        if parsed is not None:
            return _clamp(parsed)

    headers = ctx.get("headers")
    if isinstance(headers, dict):
        for header in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            parsed = parse_retry_after_seconds(headers.get(header))
            if parsed is not None:
                return _clamp(parsed)

    parsed = parse_retry_after_seconds(message)
    if parsed is not None:
        return _clamp(parsed)

    reset_at = ctx.get("reset_at")
    remaining = _remaining_from_reset_at(reset_at)
    if remaining is not None:
        return _clamp(remaining)

    from_message = _reset_at_remaining_in_message(message)
    if from_message is not None:
        return _clamp(from_message)

    return float(DEFAULT_COOLDOWN_SECONDS)


def _clamp(seconds: float) -> float:
    return float(min(MAX_COOLDOWN_SECONDS, max(MIN_COOLDOWN_SECONDS, seconds)))


def _remaining_from_reset_at(reset_at: Any) -> Optional[float]:
    """Seconds until an absolute reset stamp (epoch number or ISO string); None when unusable."""
    if reset_at is None:
        return None
    if isinstance(reset_at, (int, float)) and not isinstance(reset_at, bool):
        remaining = float(reset_at) - time.time()
        return remaining if remaining > 0 else None
    if isinstance(reset_at, str) and reset_at.strip():
        try:
            from agent.credential_pool import _parse_absolute_timestamp

            parsed = _parse_absolute_timestamp(reset_at)
        except Exception:
            parsed = None
        if parsed is None:
            return None
        remaining = float(parsed) - time.time()
        return remaining if remaining > 0 else None
    return None


def _reset_at_remaining_in_message(message: str) -> Optional[float]:
    """``(reset after 1m 33s)`` style hint from a rendered error string."""
    if not isinstance(message, str):
        return None
    match = _DURATION_TEXT_RE.search(message)
    if not match:
        return None
    hours, minutes, seconds = (float(g) for g in (match.group(1) or 0, match.group(3) or 0, match.group(5) or 0))
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def is_rate_limit_error(message: str, status_code: Optional[int] = None,
                        error_context: Optional[Dict[str, Any]] = None) -> bool:
    """True when a provider error is a 429 / free-tier throttle / model cooldown.

    ``status_code`` is honoured when the caller has it, but a rendered message
    carrying ``429`` counts too: the retry loop's ``error_msg`` is a string by then.
    """
    ctx = error_context or {}
    haystack = " ".join(
        str(part) for part in (message, ctx.get("message"), ctx.get("reason"),
                              ctx.get("code"), ctx.get("type")) if part
    ).lower()
    code = " ".join(str(part) for part in (ctx.get("code"), ctx.get("reason")) if part).lower()

    if status_code is not None and int(status_code) != 429:
        return False
    if status_code is None and "429" not in haystack:
        return False
    if any(pattern in haystack for pattern in _FREE_TIER_PATTERNS):
        return True
    if any(code_name in code for code_name in _COOLDOWN_CODES):
        return True
    return "rate limit" in haystack or "rate_limit" in haystack or "too many requests" in haystack


def arm_cooldown(provider: str, model: str, seconds: float, *, base_url: str = "",
                 reason: str = "free_rate_limited") -> Optional[Cooldown]:
    """Record a cooldown for ``(provider, model)``. Returns the armed window.

    A shorter window never overwrites a longer one still running: the provider
    telling us "90s" after having told us "600s" must not shorten the bench.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if not model:
        return None

    now = time.time()
    ttl = _clamp(float(seconds))
    state = _read_state()
    entries = _prune(state)
    key = _entry_key(provider, model, base_url)

    existing = entries.get(key)
    if isinstance(existing, dict):
        existing_remaining = _remaining_from_reset_at(existing.get("reset_at"))
        if existing_remaining is not None and existing_remaining > ttl:
            ttl = existing_remaining

    entries[key] = {
        "provider": provider,
        "model": model,
        "base_url": _normalize_base_url(base_url),
        "armed_at": now,
        "reset_at": now + ttl,
        "retry_after": ttl,
        "reason": reason,
    }
    if not _write_state(_trim(entries)):
        logger.warning("Rate-limit cooldown for %s/%s NOT persisted (sidecar unwritable)", provider, model)
        return None
    logger.info("Rate-limit cooldown armed: %s/%s for %ds (%s)", provider, model, int(ttl), reason)
    return Cooldown(provider=provider, model=model, reset_at=now + ttl, retry_after=ttl)


def _prune(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drop expired/garbage entries; ``state`` may hold legacy top-level keys."""
    now = time.time()
    entries: Dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(value, dict):
            continue
        if _remaining_from_reset_at(value.get("reset_at")) is None:
            continue
        entries[key] = value
    return entries


def _trim(entries: Dict[str, Any]) -> Dict[str, Any]:
    if len(entries) <= _MAX_ENTRIES:
        return entries
    newest = sorted(entries.items(), key=lambda kv: kv[1].get("armed_at", 0), reverse=True)
    return dict(newest[:_MAX_ENTRIES])


def active_cooldown(provider: str, model: str, base_url: str = "") -> Optional[Cooldown]:
    """The cooldown currently benching ``(provider, model)``; None when free.

    Falls back to the provider-label key so a cooldown armed before an endpoint
    was known is still honoured.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if not model:
        return None
    state = _read_state()
    entry = state.get(_entry_key(provider, model, base_url))
    if entry is None and _normalize_base_url(base_url):
        entry = state.get(_entry_key(provider, model))
    if not isinstance(entry, dict):
        return None
    remaining = _remaining_from_reset_at(entry.get("reset_at"))
    if remaining is None:
        return None
    return Cooldown(
        provider=provider,
        model=model,
        reset_at=float(entry["reset_at"]),
        retry_after=float(entry.get("retry_after") or remaining),
    )


def is_cooldown_active(provider: str, model: str, base_url: str = "") -> Tuple[bool, Optional[float]]:
    """``(active, remaining_seconds)`` for the skip check."""
    cooldown = active_cooldown(provider, model, base_url)
    if cooldown is None:
        return False, None
    return True, cooldown.remaining(time.time())


def log_cooldown_skip(provider: str, model: str, cooldown: Cooldown) -> str:
    """Emit the skip log line carrying the cooldown timestamp; return it for tests."""
    now = time.time()
    reset_clock = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(cooldown.reset_at))
    line = (
        f"Fallback skip: {provider}/{model} rate-limited (cooldown "
        f"{int(cooldown.remaining(now))}s left, {cooldown.percent(now)}% of {int(cooldown.retry_after)}s, "
        f"until {reset_clock} UTC+{time.strftime('%z')[:3]})"
    )
    logger.info(line)
    return line


def clear_cooldown(provider: str, model: str, base_url: str = "") -> None:
    """Drop a cooldown once the pair demonstrably works again."""
    state = _read_state()
    entries = _prune(state)
    if entries.pop(_entry_key(provider, model, base_url), None) is not None:
        _write_state(_trim(entries))
        logger.debug("Rate-limit cooldown cleared: %s/%s", provider, model)
