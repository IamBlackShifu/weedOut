"""Tests for the BullyingClassifier."""

import tempfile
from pathlib import Path

import pytest

from weedout.models.classifier import BullyingClassifier

_TRAIN_TEXTS = [
    "I hate you so much, you are disgusting",
    "You are so stupid and ugly, nobody likes you",
    "You are worthless and should just disappear",
    "Go away loser, you are pathetic",
    "I will make your life miserable, you freak",
    "Have a great day, sunshine!",
    "Thanks for your help, really appreciate it",
    "Looking forward to our meeting tomorrow",
    "Beautiful weather today, hope you enjoy it",
    "Great job on the project, well done!",
]
_TRAIN_LABELS = [
    "bullying", "bullying", "bullying", "bullying", "bullying",
    "non-bullying", "non-bullying", "non-bullying", "non-bullying", "non-bullying",
]


@pytest.fixture()
def fitted_clf():
    clf = BullyingClassifier(uncertainty_threshold=0.55)
    clf.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    return clf


class TestFitPredict:
    def test_fit_runs_without_error(self):
        clf = BullyingClassifier()
        clf.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
        assert clf.is_fitted

    def test_predict_returns_list(self, fitted_clf):
        results = fitted_clf.predict(["some test text"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_predict_labels_are_valid(self, fitted_clf):
        valid = {"bullying", "non-bullying", "uncertain"}
        labels = fitted_clf.predict(_TRAIN_TEXTS)
        assert all(l in valid for l in labels)

    def test_predict_one_structure(self, fitted_clf):
        result = fitted_clf.predict_one("you are so stupid")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_proba_shape(self, fitted_clf):
        probs = fitted_clf.predict_proba(_TRAIN_TEXTS)
        assert probs.shape == (len(_TRAIN_TEXTS), 2)

    def test_obvious_bullying_detected(self, fitted_clf):
        label = fitted_clf.predict(["I hate you, you disgusting worthless loser"])[0]
        assert label in ("bullying", "uncertain")

    def test_obvious_friendly_not_bullying(self, fitted_clf):
        label = fitted_clf.predict(["Great work today, have a wonderful evening!"])[0]
        assert label in ("non-bullying", "uncertain")

    def test_requires_fit_before_predict(self):
        clf = BullyingClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(["some text"])


class TestPersistence:
    def test_save_and_load(self, fitted_clf):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = Path(tmp.name)

        fitted_clf.save(path)
        loaded = BullyingClassifier.load(path)
        assert loaded.is_fitted

        original_pred = fitted_clf.predict(_TRAIN_TEXTS[:3])
        loaded_pred = loaded.predict(_TRAIN_TEXTS[:3])
        assert original_pred == loaded_pred

        path.unlink()

    def test_save_requires_fit(self):
        clf = BullyingClassifier()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
            with pytest.raises(RuntimeError):
                clf.save(tmp.name)
