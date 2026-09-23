"""Output-token window guard for a prepared API request.

``max_tokens`` is a reservation on the same context window the prompt occupies, so any
fixed cap is only safe while the prompt is short. Once ``prompt + cap`` crosses the model
window the provider rejects the whole request — vLLM/OpenRouter answer with "This model's
maximum context length is N tokens. However, you requested M output tokens ..." — and the
rejection is not repairable by compression, because the input itself fits. That is how
local-vllm (qwen3.8-27b-nvfp4, window 190000) died on 124465 input + 65536 output = 190001,
taking the bg-review and compression lanes down with it.

The guard derives the cap from the request instead of trusting a constant: an outbound cap
is lowered to what the window has left, and a prompt that already ate the room a usable
answer needs is refused before the call (a clear error instead of an HTTP 400).

Only doomed requests are touched: when the requested cap already fits the remaining
window the value is returned unchanged, so no provider sees a different cap than before.
Requests carrying no cap are left alone too — there the provider fits the answer to the
window itself, so there is nothing to clamp and nothing to reject.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

# Kept below the model window: the prompt size comes from a char-based estimate (not the
# provider tokenizer) and providers add their own prompt-template overhead on top.
OUTPUT_WINDOW_HEADROOM_TOKENS = 1024
# An answer this short is useless: refuse the request rather than send a doomed one.
MIN_OUTPUT_TOKENS = 256

# Request fields carrying an outbound output cap, most specific first.
_OUTPUT_CAP_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")


class OutputWindowExhausted(RuntimeError):
    """The prompt leaves no room for a usable answer: the request must not be sent."""


def output_cap_field(api_kwargs: Any) -> Tuple[Optional[str], Optional[int]]:
    """``(field, value)`` of the outbound output cap in a prepared request.

    ``(None, None)`` when the request carries no cap (provider default applies) — the
    field name matters to the caller because the clamp must rewrite the same key.
    """
    if not isinstance(api_kwargs, dict):
        return None, None
    for field in _OUTPUT_CAP_FIELDS:
        raw = api_kwargs.get(field)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return field, value
    return None, None


def requested_output_cap(api_kwargs: Any) -> Optional[int]:
    """Outgoing output cap of a prepared request, or ``None`` when it carries none."""
    return output_cap_field(api_kwargs)[1]


def available_output_tokens(
    context_length: int, prompt_tokens: int, *, headroom: int = OUTPUT_WINDOW_HEADROOM_TOKENS
) -> int:
    """Tokens the answer may use: ``context_length - headroom - prompt_tokens``, floored at 0."""
    return max(0, int(context_length) - int(headroom) - max(0, int(prompt_tokens)))


def clamp_output_tokens(requested: int, available: int, *, minimum: int = MIN_OUTPUT_TOKENS) -> Optional[int]:
    """``min(requested, available)``; ``None`` when not even ``minimum`` fits (do not call)."""
    available = max(0, int(available))
    if available < int(minimum):
        return None
    return min(int(requested), available)


def apply_output_window_guard(agent: Any, api_kwargs: Any, *, prompt_tokens: int) -> Optional[int]:
    """Clamp the outbound output cap so ``prompt + cap`` stays inside the model window.

    Returns the cap left on the request (``None`` when the request carries none or the
    window is unknown). Raises :class:`OutputWindowExhausted` when the prompt already
    consumed the room a usable answer needs — callers must then skip the request.
    """
    field, requested = output_cap_field(api_kwargs)
    if field is None or requested is None:
        return None
    context_length = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        return requested
    clamped = clamp_output_tokens(requested, available_output_tokens(context_length, prompt_tokens))
    if clamped is None:
        raise OutputWindowExhausted(
            f"Prompt of ~{prompt_tokens:,} tokens leaves less than {MIN_OUTPUT_TOKENS} tokens of the "
            f"{context_length:,}-token window (headroom {OUTPUT_WINDOW_HEADROOM_TOKENS}); not sending the "
            f"request. Compress the conversation or raise the model's context length."
        )
    if clamped != requested:
        api_kwargs[field] = clamped
    return clamped
