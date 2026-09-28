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


# ---------------------------------------------------------------------------
# The production wiring: detector -> arming -> walk skip.
#
# The sidecar itself is unit-covered above; these tests pin the two ROUTES
# that must reach it, because both were silent defects: the router threw the
# original body away (ar msg_str has no ``status_code`` and no ``reset_at``),
# and the walk never re-read the sidecar when config could not resolve the
# candidate's base_url.
# ---------------------------------------------------------------------------

MODEL = "anymodel/ds/deepseek-v4-flash"
SIBLING = "bai/mimo-v2.6-flash"

# 503 + free_tier_body: the shape the classification layer throws as
# ``overloaded`` (a reason try_activate_fallback never sees). Arming it is
# the whole point of the detector, not just ``is_rate_limit_error``'s call
# sites.
FREE_503_MESSAGE = (
    "Error code: 503 - {'error': {'message': 'free_rate_limited: no free quota "
    "left for this endpoint', 'type': 'rate_limit_error', 'code': "
    "'free_rate_limited', 'reset_seconds': 94}}"
)
# The same status WITHOUT a throttle body: an upstream outage is not a model
# to bench, only a transport failure to retry.
PLAIN_503_MESSAGE = "Error code: 503 - {'error': {'message': 'upstream overloaded'}}"
# The request-shape cap: it arrives as a 429 but is excluded by the budget
# check in route_classified_error (that one is why max_tokens is clamped).
WRAPPED_429_MESSAGE = (
    "Error code: 429 - {'error': {'message': '[400]: max_tokens (16384) exceeds "
    "model maximum output tokens (8192)', 'type': 'invalid_request_error'}}"
)
AUTH_401_MESSAGE = (
    "Error code: 401 - {'error': {'message': 'rate limit exceeded, check your "
    "plan', 'type': 'authentication_error'}}"
)


class _APIError(Exception):
    """The exception type the loop raises with ``status_code`` bolted on."""

    status_code = None


def _agent(message="", status=None, context=None):
    """The narrow agent surface route_classified_error + the walk touch."""
    from types import SimpleNamespace

    agent = SimpleNamespace(
        provider="llm-router",
        model=MODEL,
        base_url=ROUTER,
        requested_provider="llm-router",
        _primary_runtime={"provider": "llm-router", "model": MODEL, "base_url": ROUTER},
        _fallback_activated=False,
        _fallback_index=0,
        _fallback_chain=[],
        compression_enabled=True,
        _last_api_error_message=message,
        _last_api_error_status=status,
        _last_api_error_context=context,
    )
    return agent


def _route(agent, message, status, context=None):
    """Run the real error router over ``message``; returns its verdict."""
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import route_classified_error
    from agent.turn_retry_state import TurnRetryState

    err = _APIError(message)
    err.status_code = status
    classified = classify_api_error(err, model=MODEL, provider="llm-router", base_url=ROUTER)
    return route_classified_error(
        agent,
        err,
        classified,
        TurnRetryState(),
        error_msg=message,
        error_context=context or {},
        recovered_with_pool=False,
        base_url=ROUTER,
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        api_messages=[{"role": "user", "content": "hi"}],
        system_message=None,
        active_system_prompt=None,
        conversation_history=[],
        retry_count=0,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=3,
        api_call_count=1,
        effective_task_id=None,
    )


class TestErrorRouterArmsTheSidecar:
    """route_classified_error must bench the model for the NEXT walk too.

    A free-tier 503 classifies as ``overloaded``, so try_activate_fallback
    never runs on it; without the direct arm here every later turn re-selects
    the benched model after a restart.
    """

    def test_503_free_rate_limited_benches_the_model(self):
        agent = _agent()
        verdict = _route(agent, FREE_503_MESSAGE, 503)
        assert verdict.action == "fallthrough"
        cooldown = sidecar.active_cooldown("llm-router", MODEL, ROUTER)
        assert cooldown is not None
        assert cooldown.reset_at > time.time()

    def test_503_free_rate_limited_benches_under_both_keys(self):
        _route(_agent(), FREE_503_MESSAGE, 503)
        # The endpoint key (walk has no config) AND the provider/model label key.
        assert sidecar.active_cooldown("llm-router", MODEL, ROUTER) is not None
        assert sidecar.active_cooldown("llm-router", MODEL, "") is not None

    def test_plain_503_overload_does_not_bench(self):
        _route(_agent(), PLAIN_503_MESSAGE, 503)
        assert sidecar.active_cooldown("llm-router", MODEL, ROUTER) is None

    def test_401_prose_does_not_bench(self):
        _route(_agent(), AUTH_401_MESSAGE, 401)
        assert sidecar.active_cooldown("llm-router", MODEL, ROUTER) is None

    def test_wrapped_output_cap_429_does_not_bench(self):
        """The budget path (max_tokens clamp) is not a throttle: excluded."""
        _route(_agent(), WRAPPED_429_MESSAGE, 429)
        assert sidecar.active_cooldown("llm-router", MODEL, ROUTER) is None

    def test_429_benches_without_a_reset_hint_in_context(self):
        """reset_at lives in the body, not in error_context — still must arm."""
        _route(_agent(), LIVE_429_MESSAGE, 429)
        cooldown = sidecar.active_cooldown("llm-router", MODEL, ROUTER)
        assert cooldown is not None
        assert cooldown.reset_at > time.time()


class TestWalkSkipsTheBenchedEntry:
    """_should_skip_fallback_candidate must consult the sidecar ALWAYS.

    Every case here runs a seeker whose OWN model differs from the benched one:
    agent.backend_identity's "same backend" guard also skips an entry equal to
    the current backend, so without that separation a pass could come from the
    identity guard instead of from the sidecar — the exact reason the incident
    review asked for proof, not for another model-name comparison.
    """

    def _walk(self, seeker, fb_model=MODEL):
        from agent import chat_completion_helpers as helpers

        fb = {"provider": "llm-router", "base_url": "", "model": fb_model}
        fb_key = ("llm-router", fb_model)
        return helpers._should_skip_fallback_candidate(seeker, fb, fb_key, "llm-router", fb_model, set())

    @staticmethod
    def _seeker():
        """A walker running a healthy sibling, so identity cannot decide."""
        seeker = _agent()
        seeker.model = SIBLING
        return seeker

    @staticmethod
    def _unresolvable(seeker):
        """The walk's position during the incident: no config to resolve base_url."""
        seeker.requested_provider = "someone-else"
        seeker._primary_runtime = {}
        return seeker

    def test_unresolvable_base_url_still_skips_the_benched_model(self):
        _route(_agent(), FREE_503_MESSAGE, 503)
        seeker = self._unresolvable(self._seeker())
        from agent import chat_completion_helpers as helpers

        # The sidecar is then the ONLY signal that can name this entry: the
        # config lookup yields nothing and identity compares different models.
        assert helpers._fallback_entry_base_url(seeker, {"provider": "llm-router"}, "llm-router") == ""
        assert self._walk(seeker) is True

    def test_resolved_base_url_skips_through_the_endpoint_key(self):
        _route(_agent(), FREE_503_MESSAGE, 503)
        seeker = self._seeker()
        # _fallback_entry_base_url resolves to the primary's endpoint here, so
        # the walk hits the endpoint key the arming wrote.
        from agent import chat_completion_helpers as helpers

        assert helpers._fallback_entry_base_url(seeker, {"provider": "llm-router"}, "llm-router") == ROUTER
        assert self._walk(seeker) is True

    def test_a_sibling_is_not_skipped(self):
        _route(_agent(), FREE_503_MESSAGE, 503)
        # The primary's own model running, asking for a DIFFERENT entry: only
        # the benched (provider, model) pair may be suppressed.
        assert self._walk(_agent(), SIBLING) is False

    def test_clearing_the_cooldown_re_admits_the_entry(self):
        _route(_agent(), FREE_503_MESSAGE, 503)
        sidecar.clear_cooldown("llm-router", MODEL, ROUTER)
        sidecar.clear_cooldown("llm-router", MODEL, "")
        assert sidecar.active_cooldown("llm-router", MODEL, ROUTER) is None
        seeker = self._unresolvable(self._seeker())
        assert self._walk(seeker) is False
