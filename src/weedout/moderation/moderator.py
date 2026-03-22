"""
Moderation policy engine.

Maps user risk scores and individual post labels to enforcement actions:
  WARN          → surface a platform warning to the author.
  SHADOW_BAN    → hide the post from others without notifying the author.
  SUSPEND       → temporarily restrict the account.
  ESCALATE      → route to a human moderator for review.
  NONE          → no action required.

Policy table (evaluated in priority order):
  1. risk_level == HIGH  AND label == 'bullying'   → SUSPEND
  2. risk_level == HIGH  AND label == 'uncertain'  → ESCALATE
  3. risk_level == MEDIUM AND label == 'bullying'  → SHADOW_BAN
  4. risk_level == MEDIUM AND label == 'uncertain' → ESCALATE
  5. risk_level == LOW   AND label == 'bullying'   → WARN
  6. label == 'uncertain'                          → ESCALATE
  7. Otherwise                                     → NONE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from weedout.profiling.user_profiler import PostRecord, RiskLevel, UserProfile, UserProfileStore

Action = Literal["NONE", "WARN", "SHADOW_BAN", "SUSPEND", "ESCALATE"]


@dataclass
class ModerationDecision:
    """The outcome of evaluating a single post for moderation."""

    user_id: str
    post_id: str
    label: str
    confidence: float
    risk_level: RiskLevel
    risk_score: float
    action: Action
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "post_id": self.post_id,
            "label": self.label,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 4),
            "action": self.action,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class ModerationEngine:
    """
    Apply the moderation policy to a classified post and a user profile.

    Parameters
    ----------
    profile_store : UserProfileStore
        Shared store used to look up and update user profiles.
    """

    def __init__(self, profile_store: UserProfileStore | None = None) -> None:
        self._store = profile_store if profile_store is not None else UserProfileStore()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        user_id: str,
        post_id: str,
        text: str,
        label: str,
        confidence: float,
        targets: list[str] | None = None,
    ) -> ModerationDecision:
        """
        Record the post, update the user profile, and return a moderation decision.

        Parameters
        ----------
        user_id : str
            Pseudonymised identifier of the author.
        post_id : str
            Unique identifier of the post.
        text : str
            Raw post text (stored for audit purposes).
        label : str
            Classifier output: 'bullying' | 'non-bullying' | 'uncertain'.
        confidence : float
            Model confidence in the predicted label (0–1).
        targets : list[str] | None
            Usernames mentioned/targeted in the post.
        """
        profile = self._store.get_or_create(user_id)
        record = PostRecord(
            post_id=post_id,
            text=text,
            label=label,
            confidence=confidence,
            targets=targets or [],
        )
        profile.record_post(record)

        action, reason = self._policy(label, profile)
        return ModerationDecision(
            user_id=user_id,
            post_id=post_id,
            label=label,
            confidence=confidence,
            risk_level=profile.risk_level(),
            risk_score=profile.risk_score(),
            action=action,
            reason=reason,
        )

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self._store.get(user_id)

    def high_risk_users(self) -> list[str]:
        return [p.user_id for p in self._store.high_risk_users()]

    # ------------------------------------------------------------------
    # Policy rules
    # ------------------------------------------------------------------

    @staticmethod
    def _policy(label: str, profile: UserProfile) -> tuple[Action, str]:
        risk = profile.risk_level()

        if risk == "HIGH" and label == "bullying":
            return "SUSPEND", "Repeat high-risk offender with bullying content."
        if risk == "HIGH" and label == "uncertain":
            return "ESCALATE", "High-risk user; ambiguous content requires human review."
        if risk == "MEDIUM" and label == "bullying":
            return "SHADOW_BAN", "Medium-risk user with confirmed bullying content."
        if risk == "MEDIUM" and label == "uncertain":
            return "ESCALATE", "Medium-risk user; ambiguous content requires human review."
        if risk == "LOW" and label == "bullying":
            return "WARN", "First-time or low-risk bullying post; author warned."
        if label == "uncertain":
            return "ESCALATE", "Content is ambiguous; human review required."
        return "NONE", "Content classified as non-bullying; no action required."
