"""Tests for the user profiling / risk scoring module."""

from datetime import datetime, timedelta, timezone

import pytest

from weedout.profiling.user_profiler import PostRecord, UserProfile, UserProfileStore


def _make_post_record(label: str, confidence: float = 0.9, targets=None, age_days: float = 0) -> PostRecord:
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    return PostRecord(
        post_id="p1",
        text="some text",
        label=label,
        confidence=confidence,
        targets=targets or [],
        timestamp=ts,
    )


class TestUserProfile:
    def test_empty_profile_has_zero_risk(self):
        profile = UserProfile("user1")
        assert profile.risk_score() == 0.0
        assert profile.risk_level() == "LOW"

    def test_single_bullying_post_raises_risk(self):
        profile = UserProfile("user1")
        profile.record_post(_make_post_record("bullying", confidence=0.95))
        assert profile.risk_score() > 0.0

    def test_non_bullying_posts_keep_low_risk(self):
        profile = UserProfile("user1")
        for _ in range(5):
            profile.record_post(_make_post_record("non-bullying"))
        assert profile.risk_level() == "LOW"

    def test_many_bullying_posts_high_risk(self):
        profile = UserProfile("user1")
        for _ in range(10):
            profile.record_post(_make_post_record("bullying", confidence=0.95))
        assert profile.risk_level() in ("MEDIUM", "HIGH")

    def test_targeting_breadth_increases_score(self):
        profile_narrow = UserProfile("user1")
        profile_wide = UserProfile("user2")

        # Same number of bullying posts but different targets
        for i in range(5):
            profile_narrow.record_post(_make_post_record("bullying", targets=["@victim1"]))
            profile_wide.record_post(_make_post_record("bullying", targets=[f"@victim{i}"]))

        assert profile_wide.risk_score() >= profile_narrow.risk_score()

    def test_old_posts_decay(self):
        profile_recent = UserProfile("user1")
        profile_old = UserProfile("user2")

        # Add multiple bullying posts at different ages: recent posts should produce
        # a higher risk score than mostly-old posts when mixed with a current baseline.
        for _ in range(3):
            profile_recent.record_post(_make_post_record("bullying", age_days=1))
        for _ in range(3):
            profile_old.record_post(_make_post_record("bullying", age_days=90))
        # Also add an old post to recent and recent post to old so totals stay comparable
        profile_recent.record_post(_make_post_record("non-bullying", age_days=90))
        profile_old.record_post(_make_post_record("non-bullying", age_days=1))

        # Profile with recent bullying should score higher than profile with old bullying
        assert profile_recent.risk_score() > profile_old.risk_score()

    def test_summary_structure(self):
        profile = UserProfile("user1")
        profile.record_post(_make_post_record("bullying"))
        s = profile.summary()
        assert s["user_id"] == "user1"
        assert s["bullying_posts"] == 1
        assert "risk_score" in s
        assert "risk_level" in s


class TestUserProfileStore:
    def test_get_or_create(self):
        store = UserProfileStore()
        profile = store.get_or_create("user1")
        assert profile.user_id == "user1"
        # Second call returns the same object
        assert store.get_or_create("user1") is profile

    def test_get_returns_none_for_unknown(self):
        store = UserProfileStore()
        assert store.get("nonexistent") is None

    def test_high_risk_users(self):
        store = UserProfileStore()
        profile = store.get_or_create("bully1")
        for _ in range(15):
            profile.record_post(_make_post_record("bullying", confidence=1.0))
        high = store.high_risk_users()
        assert any(p.user_id == "bully1" for p in high)

    def test_len(self):
        store = UserProfileStore()
        store.get_or_create("a")
        store.get_or_create("b")
        assert len(store) == 2
