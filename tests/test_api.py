"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from weedout.api.app import _classifier, app

client = TestClient(app)

_TRAIN_PAYLOAD = {
    "texts": [
        "I hate you so much",
        "You are so stupid and ugly",
        "You are worthless",
        "Go away loser",
        "Have a great day",
        "Thanks for your help",
        "Looking forward to meeting",
        "Beautiful weather today",
    ],
    "labels": [
        "bullying", "bullying", "bullying", "bullying",
        "non-bullying", "non-bullying", "non-bullying", "non-bullying",
    ],
}


@pytest.fixture(autouse=True)
def ensure_trained():
    """Ensure the classifier is trained before each test."""
    client.post("/train", json=_TRAIN_PAYLOAD)
    yield


class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestClassify:
    def test_classify_returns_label(self):
        resp = client.post("/classify", json={"text": "you are so stupid"})
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
        assert data["label"] in ("bullying", "non-bullying", "uncertain")

    def test_classify_returns_probabilities(self):
        resp = client.post("/classify", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "probabilities" in data
        assert isinstance(data["probabilities"], dict)

    def test_classify_empty_text_rejected(self):
        resp = client.post("/classify", json={"text": ""})
        assert resp.status_code == 422

    def test_classifier_not_ready_503(self):
        # Temporarily unfit the classifier
        original_model = _classifier._model
        _classifier._model = None
        try:
            resp = client.post("/classify", json={"text": "hello"})
            assert resp.status_code == 503
        finally:
            _classifier._model = original_model


class TestModerate:
    def test_moderate_returns_decision(self):
        resp = client.post(
            "/moderate",
            json={"user_id": "testuser", "text": "you are so stupid"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "testuser"
        assert "action" in data
        assert "risk_level" in data

    def test_moderate_auto_generates_post_id(self):
        resp = client.post(
            "/moderate",
            json={"user_id": "testuser2", "text": "hello world"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["post_id"] is not None and data["post_id"] != ""

    def test_moderate_accepts_targets(self):
        resp = client.post(
            "/moderate",
            json={
                "user_id": "testuser3",
                "text": "you are stupid",
                "targets": ["@victim1"],
            },
        )
        assert resp.status_code == 200


class TestProfile:
    def test_get_profile_after_moderate(self):
        client.post("/moderate", json={"user_id": "profiled_user", "text": "test text"})
        resp = client.get("/profile/profiled_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "profiled_user"
        assert "risk_level" in data

    def test_get_profile_unknown_user_404(self):
        resp = client.get("/profile/ghost_user_xyz")
        assert resp.status_code == 404


class TestHighRiskUsers:
    def test_high_risk_users_returns_list(self):
        resp = client.get("/high-risk-users")
        assert resp.status_code == 200
        data = resp.json()
        assert "high_risk_users" in data
        assert isinstance(data["high_risk_users"], list)


class TestTrain:
    def test_train_endpoint(self):
        resp = client.post("/train", json=_TRAIN_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "trained"
        assert data["num_samples"] == len(_TRAIN_PAYLOAD["texts"])

    def test_train_mismatched_lengths_rejected(self):
        resp = client.post(
            "/train",
            json={"texts": ["a", "b"], "labels": ["bullying"]},
        )
        assert resp.status_code == 422

    def test_train_invalid_labels_rejected(self):
        resp = client.post(
            "/train",
            json={"texts": ["a", "b"], "labels": ["bullying", "invalid_label"]},
        )
        assert resp.status_code == 422
