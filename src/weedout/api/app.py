"""
FastAPI application exposing the cyberbullying detection and moderation pipeline.

Endpoints
---------
POST /classify
    Classify a single post and return label + probabilities.

POST /moderate
    Classify a post, update the author's profile, and return a moderation decision.

GET  /profile/{user_id}
    Retrieve the risk profile summary for a user.

GET  /high-risk-users
    List all users currently at HIGH risk level.

GET  /health
    Simple liveness probe.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from weedout.models.classifier import BullyingClassifier
from weedout.moderation.moderator import ModerationEngine
from weedout.profiling.user_profiler import UserProfileStore

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

_store = UserProfileStore()
_classifier = BullyingClassifier()
_engine = ModerationEngine(profile_store=_store)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Nothing to initialise at startup for the in-memory demo.
    # A real deployment would load a pre-trained model from disk here:
    #   _classifier = BullyingClassifier.load("models/classifier.pkl")
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WeedOut – Cyberbullying Detection API",
    description=(
        "NLP-powered API for detecting, profiling, and moderating cyberbullying "
        "on Twitter-like social media platforms."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=5000, description="Raw post text")]


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModerateRequest(BaseModel):
    user_id: Annotated[str, Field(description="Pseudonymised author identifier")]
    text: Annotated[str, Field(min_length=1, max_length=5000, description="Raw post text")]
    post_id: Annotated[
        str | None,
        Field(default=None, description="Optional post identifier; auto-generated if omitted"),
    ] = None
    targets: Annotated[
        list[str],
        Field(description="Usernames mentioned or targeted in the post"),
    ] = []


class ModerateResponse(BaseModel):
    user_id: str
    post_id: str
    label: str
    confidence: float
    risk_level: str
    risk_score: float
    action: str
    reason: str
    timestamp: str


class ProfileResponse(BaseModel):
    user_id: str
    total_posts: int
    bullying_posts: int
    uncertain_posts: int
    non_bullying_posts: int
    unique_targets: int
    risk_score: float
    risk_level: str


# ---------------------------------------------------------------------------
# Dependency: ensure classifier is fitted
# ---------------------------------------------------------------------------


def get_classifier() -> BullyingClassifier:
    if not _classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classifier has not been trained yet. "
                "POST training data to /train or load a pre-trained model."
            ),
        )
    return _classifier


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "classifier_ready": _classifier.is_fitted}


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
def classify(
    req: ClassifyRequest,
    clf: Annotated[BullyingClassifier, Depends(get_classifier)],
) -> ClassifyResponse:
    """Classify a single post and return a label with probabilities."""
    result = clf.predict_one(req.text)
    return ClassifyResponse(**result)


@app.post("/moderate", response_model=ModerateResponse, tags=["moderation"])
def moderate(
    req: ModerateRequest,
    clf: Annotated[BullyingClassifier, Depends(get_classifier)],
) -> ModerateResponse:
    """
    Classify a post, update the author's risk profile, and return a moderation decision.
    """
    result = clf.predict_one(req.text)
    post_id = req.post_id if req.post_id is not None else str(uuid.uuid4())
    decision = _engine.evaluate(
        user_id=req.user_id,
        post_id=post_id,
        text=req.text,
        label=result["label"],
        confidence=result["confidence"],
        targets=req.targets,
    )
    return ModerateResponse(**decision.as_dict())


@app.get("/profile/{user_id}", response_model=ProfileResponse, tags=["profiling"])
def get_profile(user_id: str) -> ProfileResponse:
    """Retrieve the risk profile for a specific user."""
    profile = _store.get(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for user '{user_id}'.")
    return ProfileResponse(**profile.summary())


@app.get("/high-risk-users", tags=["profiling"])
def high_risk_users() -> dict:
    """List all users currently flagged as HIGH risk."""
    return {"high_risk_users": _engine.high_risk_users()}


# ---------------------------------------------------------------------------
# Training endpoint (demo / admin use)
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    texts: list[str] = Field(min_length=2, description="Training post texts")
    labels: list[str] = Field(min_length=2, description="Corresponding labels ('bullying' or 'non-bullying')")


@app.post("/train", tags=["admin"])
def train(req: TrainRequest) -> dict:
    """
    Train (or retrain) the classifier on the provided data.

    This endpoint is intended for development and demonstration purposes.
    In production, model training should happen offline with proper validation.
    """
    if len(req.texts) != len(req.labels):
        raise HTTPException(status_code=422, detail="texts and labels must have the same length.")
    valid_labels = {"bullying", "non-bullying"}
    invalid = [l for l in req.labels if l not in valid_labels]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Invalid labels: {invalid}. Must be one of {valid_labels}.")
    _classifier.fit(req.texts, req.labels)
    return {"status": "trained", "num_samples": len(req.texts)}
