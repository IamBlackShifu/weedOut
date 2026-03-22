"""
User profiling and risk scoring for cyberbullying detection.

Each ``UserProfile`` maintains an in-memory record of a user's post history,
aggregating statistics used to compute a time-decayed risk score.

Risk score (0–1):
  - Weighted combination of bullying-post ratio, severity, targeting breadth,
    and recency of violations.
  - Score ≥ 0.8  → HIGH risk
  - Score ≥ 0.5  → MEDIUM risk
  - Score < 0.5  → LOW risk
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

_UNCERTAIN_CONTRIBUTION: float = 0.30
"""Fraction of a post's weight that uncertain posts contribute to bullying score.
A value of 0.30 treats uncertain content as moderately suspicious, ensuring
ambiguous posts raise the risk score without penalising as heavily as confirmed
bullying. Adjust toward 0.0 (ignore) or 1.0 (treat as bullying) based on
empirical calibration against human-annotated data.
"""

_BREADTH_BONUS_MAX: float = 0.20
"""Maximum risk-score bonus granted for harassing a large number of unique targets.
Caps at 0.20 to avoid the bonus dominating the primary content-based score.
"""

_BREADTH_BONUS_PER_TARGET: float = 0.02
"""Score bonus applied per unique target harassed.
With the cap above, the bonus reaches its maximum after 10 unique victims.
"""


# Half-life in days for the time-decay function (violations become less
# influential after this many days)
_HALF_LIFE_DAYS: float = 30.0
_DECAY_LAMBDA: float = math.log(2) / _HALF_LIFE_DAYS


@dataclass
class PostRecord:
    """A single post associated with a user profile."""

    post_id: str
    text: str
    label: str  # 'bullying' | 'non-bullying' | 'uncertain'
    confidence: float
    targets: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfile:
    """
    Maintains the moderation history for one user.

    Parameters
    ----------
    user_id : str
        Pseudonymised user identifier.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._posts: list[PostRecord] = []

    # ------------------------------------------------------------------
    # Record keeping
    # ------------------------------------------------------------------

    def record_post(self, record: PostRecord) -> None:
        """Append a classified post to this user's history."""
        self._posts.append(record)

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def risk_score(self) -> float:
        """
        Compute a time-decayed risk score in [0, 1].

        The score is a weighted sum of per-post contributions:
          - Bullying posts with high confidence contribute the most.
          - Uncertain posts contribute a fractional amount.
          - All contributions are decayed by post age.
        """
        if not self._posts:
            return 0.0

        now = datetime.now(timezone.utc)
        total_weight = 0.0
        bullying_weight = 0.0

        for post in self._posts:
            age_days = (now - post.timestamp).total_seconds() / 86_400
            decay = math.exp(-_DECAY_LAMBDA * age_days)

            post_weight = decay
            total_weight += post_weight

            if post.label == "bullying":
                bullying_weight += post_weight * post.confidence
            elif post.label == "uncertain":
                bullying_weight += post_weight * _UNCERTAIN_CONTRIBUTION

        if total_weight == 0:
            return 0.0

        raw = bullying_weight / total_weight

        # Targeting breadth bonus: penalise users who harass many unique targets
        unique_targets = len(self._unique_targets())
        breadth_bonus = min(_BREADTH_BONUS_MAX, unique_targets * _BREADTH_BONUS_PER_TARGET)

        return min(1.0, raw + breadth_bonus)

    def risk_level(self) -> RiskLevel:
        score = self.risk_score()
        if score >= 0.8:
            return "HIGH"
        if score >= 0.5:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a human-readable summary of this user's profile."""
        total = len(self._posts)
        bullying_count = sum(1 for p in self._posts if p.label == "bullying")
        uncertain_count = sum(1 for p in self._posts if p.label == "uncertain")
        unique_targets = self._unique_targets()

        return {
            "user_id": self.user_id,
            "total_posts": total,
            "bullying_posts": bullying_count,
            "uncertain_posts": uncertain_count,
            "non_bullying_posts": total - bullying_count - uncertain_count,
            "unique_targets": len(unique_targets),
            "risk_score": round(self.risk_score(), 4),
            "risk_level": self.risk_level(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unique_targets(self) -> set[str]:
        targets: set[str] = set()
        for post in self._posts:
            targets.update(post.targets)
        return targets


class UserProfileStore:
    """
    In-memory registry of all known user profiles.

    In a production deployment this would be backed by a persistent store
    (e.g., PostgreSQL, Redis) but an in-memory dict is used here for
    simplicity and testability.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get_or_create(self, user_id: str) -> UserProfile:
        """Return the existing profile for *user_id* or create a new one."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id)
        return self._profiles[user_id]

    def get(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def all_profiles(self) -> list[UserProfile]:
        return list(self._profiles.values())

    def high_risk_users(self) -> list[UserProfile]:
        return [p for p in self._profiles.values() if p.risk_level() == "HIGH"]

    def __len__(self) -> int:
        return len(self._profiles)
