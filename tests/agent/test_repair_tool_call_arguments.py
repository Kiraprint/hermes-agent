"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json
import logging

import pytest

from agent.message_sanitization import (
    _repair_escaped_structural_quotes,
    _repair_python_literals,
    _repair_tool_call_arguments,
)
class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"



    # -- Stage 2: Python None literal --



    # -- Stage 3: trailing comma repair --


    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}


    # -- Stage 4: unclosed brackets --



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    def test_unrepairable_garbage_returns_empty_object(self):
        # No JSON structure to reconstruct: brackets/closing quotes cannot help.
        assert _repair_tool_call_arguments("garbage no json", "t") == "{}"

    def test_braces_inside_string_values_do_not_skew_the_balance(self):
        # A "}" inside a value must not be counted as closing the object: naive counting
        # sees 2 "}"-worth of closes for 1 "{" and drops a repairable call to "{}".
        result = _repair_tool_call_arguments('{"code": "}", "x": 1', "t")
        assert json.loads(result) == {"code": "}", "x": 1}

    def test_truncated_nested_array_closes_in_stack_order(self):
        # {"items": [{"n": 1}, {"n": 2 needs "}]} appended (stack order), not "}}" —
        # count-based appending grouped all braces before all brackets and never parsed.
        result = _repair_tool_call_arguments('{"items": [{"n": 1}, {"n": 2', "t")
        assert json.loads(result) == {"items": [{"n": 1}, {"n": 2}]}

    # -- Balanced but misnested: the "]" of an array of objects dropped, a "}" closing in
    # its place (#115061, deepseek-v4-flash via a portal). Counts balance, so nothing can be
    # appended; the missing closer has to be inserted BEFORE the misplaced one. --

    @pytest.mark.parametrize("raw, expected", [
        ('{"a": [{"b": 1}, {"c": 2}}]}', {"a": [{"b": 1}, {"c": 2}]}),
        ('{"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}}',
         {"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}]}),
        ('{"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}}}}',
         {"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}]}}),
        ('{"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}}',
         {"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}),
        ('{"a": [1, 2}', {"a": [1, 2]}),
    ])
    def test_misnested_closer_is_inserted_before_the_misplaced_one(self, raw, expected):
        assert json.loads(_repair_tool_call_arguments(raw, "t")) == expected

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --


class TestLoggedUnrepairablePayload:
    """Regression for the 2026-09-18 kanban_complete drop (errors.log.1:10279).

    The logged argument string was 750 bytes: a Python ``False``, the
    ``specific_pattern_searched`` value sent one escape level too deep
    (``"key": \"value\"``), an excess closing brace and a stray ``]`` — every one of
    them a defect the earlier passes cannot express, so the call was dropped as
    ``"{}"`` and kanban_complete ran with no arguments at all.
    """

    LOGGED_PAYLOAD = (
        '{"task_id": "t_0090c345", "summary": "Completed incident evidence extraction from errors.log. Produced comprehensive report showing the specific subagent-1 HTTP 503 pattern does not exist in the log. Total ERROR lines in last hour: 84, all are streaming failures from various providers. Report saved as incident_report.txt in workspace.", "metadata": {"total_errors_last_hour": 84, "subagent_503_pattern_found": False, "specific_pattern_searched": \\"[subagent-1] API call failed after 3 retries. HTTP 503\\", "error_patterns_found": 84, "unique_error_messages": 84, "warning_info_context_lines": 70, "report_file_path": "/opt/data/kanban/boards/factory/workspaces/t_0090c345/incident_report.txt", "completion_timestamp": "2026-09-18 00:10:37 UTC"}}}\n]'
    )

    def test_logged_payload_is_repaired_instead_of_dropped(self):
        repaired = _repair_tool_call_arguments(self.LOGGED_PAYLOAD, "kanban_complete")
        parsed = json.loads(repaired)  # "{}" (the old drop) parses too, keys below are the contract
        assert parsed["task_id"] == "t_0090c345"
        assert parsed["summary"].startswith("Completed incident evidence extraction")
        assert parsed["metadata"]["subagent_503_pattern_found"] is False
        assert parsed["metadata"]["specific_pattern_searched"] == (
            "[subagent-1] API call failed after 3 retries. HTTP 503"
        )
        assert parsed["metadata"]["report_file_path"].endswith("t_0090c345/incident_report.txt")

    def test_escaped_quotes_inside_string_values_are_untouched(self):
        """The structural-quote pass must never rewrite a real ``\\"`` escape in quoted content."""
        valid = '{"summary": "he said \\"hi\\" to None, loudly", "subagent_503_pattern_found": false}'
        assert _repair_python_literals(valid) == valid
        assert _repair_escaped_structural_quotes(valid) == valid

    def test_dropped_arguments_warning_carries_trace_context(self, caplog):
        """An unrepairable drop must be correlatable: session/tool-call id + the whole string."""
        truncated = '{"task_id": "t_0090c345", "summary": "half a summary'
        with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
            result = _repair_tool_call_arguments(
                truncated, "kanban_complete",
                session_id="20260918_000655_2a2f52", tool_call_id="call_abc123",
            )
        assert result == "{}"
        assert "Unrepairable tool_call arguments for kanban_complete" in caplog.text
        assert "session_id=20260918_000655_2a2f52" in caplog.text
        assert "tool_call_id=call_abc123" in caplog.text
        assert truncated in caplog.text


