"""Recover tool-call intent that leaked into assistant *content* as XML/DSML markup.

DeepSeek chat models (deepseek-v4-flash / v4-pro) intermittently "fall through" to text
mode and emit a tool invocation as literal markup inside ``message.content`` instead of the
structured ``tool_calls`` field, returning ``finish_reason="stop"`` and ``tool_calls=null``
(deepseek-ai/DeepSeek-V3#1244, #1678). Hermes only executes structured ``tool_calls``, so the
invocation is silently stored as the final answer and no tool runs.

This module detects that markup and turns it back into OpenAI-style tool calls. It handles:

* the plain form ``<tool_calls><invoke name=...><parameter name=...>...</parameter></invoke>``
* the DeepSeek internal DSML envelope ``<\uff5cDSML\uff5ctool_calls>`` and ``\uff5c`` variants,
  which defeats the literal ``strip_think_blocks`` regexes (that's why it survives to the archive)
* the wrapped-args shape ``<parameter name="arguments">{"command": ...}</parameter>``

It only fires when the agent actually has structured tool calls absent AND the content truly
contains a ``<invoke>`` block; prose that merely mentions ``<invoke>`` with no args is untouched.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any

# Envelope characters DeepSeek inserts around internal tags: U+FF5C FULLWIDTH VERTICAL LINE,
# U+2039/U+203A.
_ENV_CHARS = "\uff5c\u2039\u203a"
_ENV_CLASS = f"[{_ENV_CHARS}]*"
# _strip_env:  <\uff5cDSML\uff5ctool_calls>  or  <\uff5ctool_calls>  ->  <tool_calls>
_DSML_OPEN = re.compile(rf"<{_ENV_CLASS}(?:DSML)?{_ENV_CLASS}(?=[A-Za-z_])")
_DSML_CLOSE = re.compile(rf"</{_ENV_CLASS}(?:DSML)?{_ENV_CLASS}(?=[A-Za-z_])")

_INVOKE_RE = re.compile(r"<invoke\s+name=\"([^\"]+)\"\s*>(.*?)</invoke>", re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(
    r"<parameter\s+name=\"([^\"]+)\"\s*[^>]*?>(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)
# The enclosing block (plain or DSML-enveloped), stripped after we extract the invokes.
_WRAPPED_BLOCK_RE = re.compile(
    rf"<{_ENV_CLASS}(?:DSML)?{_ENV_CLASS}(?:tool_calls?|calls)\b.*?"
    rf"</{_ENV_CLASS}(?:DSML)?{_ENV_CLASS}(?:tool_calls?|calls)>",
    re.DOTALL | re.IGNORECASE,
)
_EXTRA_BLANK_RE = re.compile(r"\n{3,}")


def _strip_env(s: str) -> str:
    return _DSML_OPEN.sub("<", _DSML_CLOSE.sub("</", s))


def _coerce_value(v: str) -> Any:
    v = html.unescape(v.strip())
    try:
        return json.loads(v)
    except Exception:
        return v


def recover_leaked_tool_calls(
    content: Any,
    valid_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(tool_calls, cleaned_content)``.

    ``tool_calls`` are OpenAI-shaped dicts ``{"name", "arguments"}`` (arguments is a JSON string)
    suitable for the transport ``ToolCall`` constructor. ``cleaned_content`` has the consumed
    markup removed and is safe to store/display. No markup (or no call whose name is in
    ``valid_names`` when provided) → ``([], content)`` unchanged.
    """
    if not isinstance(content, str) or not content.strip():
        return [], content
    norm = _strip_env(content)
    if "<invoke" not in norm or "</invoke>" not in norm:
        return [], content
    wrapped = _WRAPPED_BLOCK_RE.search(norm) is not None
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for m in _INVOKE_RE.finditer(norm):
        name = m.group(1).strip()
        body = m.group(2)
        params = [(k.strip(), _coerce_value(v)) for k, v in _PARAM_RE.findall(body)]
        if not name:
            continue
        # False-positive guard: prose that merely references <invoke> with no args, and not
        # inside a <tool_calls> wrapper, is not a tool call.
        if not wrapped and not params:
            continue
        # Wrapped-args shape: a single parameter literally named 'arguments' whose value is
        # itself the JSON arguments object (the deepseek-v4-pro fall-through form).
        args: Any = None
        for k, v in params:
            if k.lower() == "arguments":
                args = v
                break
        if args is None:
            args = dict(params)
        if isinstance(args, str):
            # A bare-string arguments (e.g. an unquoted value) — wrap it.
            args = {"arguments": args} if not args.strip().startswith("{") else args
        if not isinstance(args, dict):
            args = {"value": args}
        if valid_names is not None and name not in valid_names:
            continue
        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
        spans.append(m.span())
    if not calls:
        return [], content
    # Prefer removing the enclosing <tool_calls> block; else remove the invokes themselves.
    cleaned = norm
    w = _WRAPPED_BLOCK_RE.search(cleaned)
    if w:
        cleaned = cleaned[: w.start()] + cleaned[w.end():]
    else:
        for start, end in sorted(spans, reverse=True):
            cleaned = cleaned[:start] + cleaned[end:]
    cleaned = _EXTRA_BLANK_RE.sub("\n\n", cleaned).strip()
    return calls, cleaned


def build_message_tool_calls(calls: list[dict[str, Any]]) -> list[Any]:
    """Wrap recovered dicts in the transport ``ToolCall`` objects the loop dispatches.

    Ids use hermes' own deterministic scheme (``message_sanitization.deterministic_call_id``)
    so multi-call batches never collide under ``_uniquify_tool_call_ids`` and prompt-cache
    prefixes stay byte-identical to what the loop would have synthesized."""
    from agent.transports.types import build_tool_call
    from agent.message_sanitization import deterministic_call_id

    built = []
    for index, c in enumerate(calls):
        name = c["name"]
        arguments = c["arguments"]
        call_id = deterministic_call_id(name, arguments, index)
        built.append(build_tool_call(id=call_id, name=name, arguments=arguments))
    return built
