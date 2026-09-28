"""Invariant tests for the per-entry rate-limit cooldown sidecar.

Regression for the live 2026-09-28 incident: on a 429 the fallback walk re-selected
the very model the router had just benched (``Fallback activated: anymodel/ds/
deepseek-v4-flash → anymodel/ds/deepseek-v4-flash``), because the chain's first entry
carries the same provider+model as the primary and the skip check only compared
provider+base_url identity — not the model that 429'd.

The contract these tests pin:
  * a 429 records a cooldown for the (endpoint, model) that failed;
  * while that window runs, the walk SKIPS that entry and reaches a healthy sibling;
  * the identity is the ENDPOINT, not the provider label, because one router answers
    to several config aliases (``custom`` and ``llm-router`` here);
  * the sidecar is a file, so the cooldown survives a process restart;
  * nothing about the sidecar may raise into the retry loop or the gateway.
"""

import json
import time

import pytest

from agent import fallback_rate_limit_sidecar as sidecar

# The real production body, verbatim from /opt/data/logs/agent.log.
LIVE_429_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'All credentials for model "
    "ds/deepseek-v4-flash are cooling down', 'type': 'rate_limit_error', "
    "'code': 'model_cooldown', 'model': 'ds/deepseek-v4-flash', 'reset_seconds': 94, "
    "'retry_after': '2026-09-28T15:00:02.816Z', 'credentials_cooling': 1}}"
)
ROUTER = "http://127.0.0.1:20130/v1"


@pytest.fixture(autouse=True)
def _isolated_sidecar(tmp_path, monkeypatch):
    """Point HERMES_HOME at the temp dir so no test touches a real profile sidecar."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestArmAndRead:
    def test_429_arms_a_cooldown_for_the_failing_model(self):
        sidecar.arm_cooldown("llm-router", "anymodel/ds/deepseek-v4-flash", 300.0, base_url=ROUTER)
        active, remaining = sidecar.is_cooldown_active(
            "llm-router", "anymodel/ds/deepseek-v4-flash", ROUTER)
        assert active is True
        assert 0 < remaining <= 300

    def test_cooldown_survives_a_process_restart(self):
        """Durability is the point of a sidecar: a new reader sees the armed window."""
        sidecar.arm_cooldown("llm-router", "m/1", 300.0, base_url=ROUTER)
        # A fresh module-level read (no in-memory cache anywhere) is the restart proxy.
        reloaded = json.loads((sidecar._sidecar_path()).read_text())
        assert any(entry.get("model") == "m/1" for entry in reloaded.values())

    def test_identity_is_the_endpoint_not_the_provider_label(self):
        """One router answers to several aliases; the cooldown must match across them."""
        sidecar.arm_cooldown("custom", "anymodel/ds/deepseek-v4-flash", 300.0, base_url=ROUTER)
        active, _ = sidecar.is_cooldown_active(
            "llm-router", "anymodel/ds/deepseek-v4-flash", "http://127.0.0.1:20130/v1/")
        assert active is True

    def test_other_models_on_the_same_endpoint_stay_eligible(self):
        sidecar.arm_cooldown("llm-router", "anymodel/ds/deepseek-v4-flash", 300.0, base_url=ROUTER)
        active, _ = sidecar.is_cooldown_active("llm-router", "bai/mimo-v2.6-flash", ROUTER)
        assert active is False

    def test_shorter_hint_never_shortens_a_running_cooldown(self):
        sidecar.arm_cooldown("llm-router", "m/1", 600.0, base_url=ROUTER)
        sidecar.arm_cooldown("llm-router", "m/1", 300.0, base_url=ROUTER)
        cooldown = sidecar.active_cooldown("llm-router", "m/1", ROUTER)
        assert cooldown.retry_after > 500

    def test_expired_entry_no_longer_benches_the_model(self):
        sidecar.arm_cooldown("llm-router", "m/1", 300.0, base_url=ROUTER)
        state = json.loads(sidecar._sidecar_path().read_text())
        for entry in state.values():
            entry["reset_at"] = time.time() - 1
        sidecar._sidecar_path().write_text(json.dumps(state))
        active, remaining = sidecar.is_cooldown_active("llm-router", "m/1", ROUTER)
        assert active is False
        assert remaining is None


class TestParseRetryAfter:
    @pytest.mark.parametrize("text,expected", [
        ("94", 94.0),
        ("retry_after: 300", 300.0),
        ("resets_in_seconds: 94", 94.0),
        ("reset_seconds: 94", 94.0),
        ("reset after 5 s", 5.0),
        ("reset after 1m", 60.0),
        ("(reset after 1m 33s)", 93.0),
        ("(reset after 3s)", 3.0),
        ("2 min", 120.0),
    ])
    def test_durations_parse_to_seconds(self, text, expected):
        assert sidecar.parse_retry_after_seconds(text) == expected

    def test_absolute_timestamp_is_not_read_as_a_relative_duration(self):
        """'retry_after': '2026-09-28T...' must not become 2026 seconds."""
        parsed = sidecar.parse_retry_after_seconds("retry_after: '2026-09-28T15:00:02.816Z'")
        assert parsed is None or parsed < 600

    def test_unparseable_input_yields_none(self):
        for text in ("", "garbage", None, "no numbers here"):
            assert sidecar.parse_retry_after_seconds(text) is None


class TestCooldownFromError:
    def test_live_429_body_clamps_into_the_configured_band(self):
        seconds = sidecar.cooldown_seconds_from_error(LIVE_429_MESSAGE)
        assert sidecar.MIN_COOLDOWN_SECONDS <= seconds <= sidecar.MAX_COOLDOWN_SECONDS

    def test_absent_hint_falls_back_to_the_default(self):
        assert sidecar.cooldown_seconds_from_error("HTTP 429: slow down") == \
            float(sidecar.DEFAULT_COOLDOWN_SECONDS)

    def test_error_context_reset_at_is_honoured(self):
        seconds = sidecar.cooldown_seconds_from_error("", {"reset_at": time.time() + 450})
        assert 400 < seconds <= sidecar.MAX_COOLDOWN_SECONDS


class TestRateLimitRecognition:
    def test_live_429_body_is_recognised(self):
        assert sidecar.is_rate_limit_error(LIVE_429_MESSAGE, 429, {"code": "model_cooldown"})

    def test_free_tier_throttle_is_recognised(self):
        assert sidecar.is_rate_limit_error(
            "429 free_rate_limited: no free quota left", 429, {"reason": "free_rate_limited"})

    @pytest.mark.parametrize("status,ctx", [(500, {}), (401, {}), (400, {})])
    def test_other_failures_are_not_rate_limits(self, status, ctx):
        assert sidecar.is_rate_limit_error("upstream broke", status, ctx) is False


class TestFailOpen:
    """A corrupt, truncated or unwritable sidecar must degrade, never raise."""

    @pytest.mark.parametrize("payload", ["{not json", '["a list"]', "", '{"k": {"reset_at": "x"}}'])
    def test_malformed_sidecar_reads_as_no_cooldown(self, payload):
        sidecar._sidecar_path().write_text(payload)
        active, remaining = sidecar.is_cooldown_active("llm-router", "m/1", ROUTER)
        assert active is False
        assert remaining is None

    def test_malformed_sidecar_is_repaired_by_the_next_arm(self):
        sidecar._sidecar_path().write_text("{not json")
        armed = sidecar.arm_cooldown("llm-router", "m/1", 300.0, base_url=ROUTER)
        assert armed is not None
        assert sidecar.is_cooldown_active("llm-router", "m/1", ROUTER)[0] is True

    def test_unwritable_home_reports_an_unarmed_cooldown(self, tmp_path, monkeypatch):
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        monkeypatch.setenv("HERMES_HOME", str(blocked))
        assert sidecar.arm_cooldown("llm-router", "m/1", 300.0, base_url=ROUTER) is None
        assert sidecar.is_cooldown_active("llm-router", "m/1", ROUTER) == (False, None)

    def test_empty_model_is_rejected_without_writing(self):
        assert sidecar.arm_cooldown("llm-router", "", 300.0, base_url=ROUTER) is None
