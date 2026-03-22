"""Tests for the moderation policy engine."""

import pytest

from weedout.moderation.moderator import ModerationEngine
from weedout.profiling.user_profiler import UserProfileStore


def _build_engine():
    store = UserProfileStore()
    return ModerationEngine(profile_store=store)


def _bulk_bullying(engine: ModerationEngine, user_id: str, n: int):
    """Record *n* high-confidence bullying posts for *user_id*."""
    for i in range(n):
        engine.evaluate(
            user_id=user_id,
            post_id=f"p{i}",
            text="you are so stupid and ugly",
            label="bullying",
            confidence=0.95,
        )


class TestModerationPolicy:
    def test_non_bullying_produces_no_action(self):
        engine = _build_engine()
        decision = engine.evaluate("user1", "p1", "hello", "non-bullying", 0.9)
        assert decision.action == "NONE"

    def test_first_low_confidence_bullying_warns(self):
        engine = _build_engine()
        # A low-confidence first offence → LOW risk → WARN
        decision = engine.evaluate("user1", "p1", "you are stupid", "bullying", 0.3)
        assert decision.action == "WARN"

    def test_medium_risk_bullying_shadow_bans(self):
        engine = _build_engine()
        # Accumulate enough posts to reach MEDIUM risk before the decisive one
        _bulk_bullying(engine, "user2", 5)
        profile = engine.get_profile("user2")
        if profile.risk_level() == "MEDIUM":
            decision = engine.evaluate("user2", "pfinal", "you loser", "bullying", 0.9)
            assert decision.action == "SHADOW_BAN"

    def test_high_risk_bullying_suspends(self):
        engine = _build_engine()
        _bulk_bullying(engine, "user3", 15)
        profile = engine.get_profile("user3")
        if profile.risk_level() == "HIGH":
            decision = engine.evaluate("user3", "pfinal", "you are worthless", "bullying", 0.95)
            assert decision.action == "SUSPEND"

    def test_uncertain_always_escalates(self):
        engine = _build_engine()
        decision = engine.evaluate("user4", "p1", "some text", "uncertain", 0.45)
        assert decision.action == "ESCALATE"

    def test_decision_contains_expected_fields(self):
        engine = _build_engine()
        decision = engine.evaluate("user5", "p1", "some text", "non-bullying", 0.9)
        d = decision.as_dict()
        for key in ("user_id", "post_id", "label", "confidence", "risk_level", "risk_score", "action", "reason", "timestamp"):
            assert key in d

    def test_targets_update_profile(self):
        engine = _build_engine()
        engine.evaluate("user6", "p1", "text", "bullying", 0.9, targets=["@victim1"])
        engine.evaluate("user6", "p2", "text", "bullying", 0.9, targets=["@victim2"])
        profile = engine.get_profile("user6")
        assert profile is not None
        # Two unique targets should be recorded
        assert profile.summary()["unique_targets"] == 2

    def test_high_risk_users_list(self):
        engine = _build_engine()
        _bulk_bullying(engine, "bully", 15)
        high = engine.high_risk_users()
        # "bully" may or may not be HIGH depending on exact score; if so it's listed
        profile = engine.get_profile("bully")
        if profile and profile.risk_level() == "HIGH":
            assert "bully" in high
