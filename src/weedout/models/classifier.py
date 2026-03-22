"""
Cyberbullying classifier.

Labels: 'bullying' | 'non-bullying' | 'uncertain'

The classifier uses a calibrated LinearSVC so it can emit probability
estimates.  The 'uncertain' label is assigned when the predicted probability
of the winning class falls below a configurable threshold.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from weedout.features.feature_extractor import FeatureExtractor

Label = Literal["bullying", "non-bullying", "uncertain"]

_BINARY_LABELS = ("non-bullying", "bullying")


class BullyingClassifier:
    """
    Train and run a cyberbullying classifier.

    The model is a TF-IDF + structural-feature pipeline feeding a calibrated
    LinearSVC.  Once trained it can be persisted to / loaded from disk.

    Parameters
    ----------
    uncertainty_threshold : float
        If the confidence of the top predicted class is below this value the
        post is labelled 'uncertain' and routed to human review.
    ngram_range : tuple[int, int]
        N-gram range forwarded to the underlying FeatureExtractor.
    max_features : int | None
        Vocabulary limit forwarded to the underlying FeatureExtractor.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.60,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 10_000,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._extractor = FeatureExtractor(
            ngram_range=ngram_range,
            max_features=max_features,
        )
        self._model: CalibratedClassifierCV | None = None
        self._classes: list[str] = list(_BINARY_LABELS)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[str]) -> "BullyingClassifier":
        """
        Train on *texts* / *labels* pairs.

        Parameters
        ----------
        texts : list[str]
            Raw post strings.
        labels : list[str]
            One of ``'bullying'`` or ``'non-bullying'`` per text.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        X = self._extractor.fit_transform(texts)
        base = LinearSVC(max_iter=2000, dual="auto")
        self._model = CalibratedClassifierCV(base, cv=min(5, len(set(labels))))
        self._model.fit(X, labels)
        self._classes = list(self._model.classes_)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> list[Label]:
        """
        Classify each text and return a list of labels.

        Posts with low confidence are labelled ``'uncertain'``.
        """
        probs = self.predict_proba(texts)
        results: list[Label] = []
        for prob_row in probs:
            best_idx = int(np.argmax(prob_row))
            best_prob = prob_row[best_idx]
            if best_prob < self.uncertainty_threshold:
                results.append("uncertain")
            else:
                results.append(self._classes[best_idx])  # type: ignore[arg-type]
        return results

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return calibrated class probabilities for each text.

        Returns an (n, 2) array where columns correspond to ``self.classes_``.
        """
        self._require_fitted()
        X = self._extractor.transform(texts)
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    def predict_one(self, text: str) -> dict:
        """
        Classify a single post and return a rich result dict.

        Returns
        -------
        dict with keys: 'label', 'confidence', 'probabilities'
        """
        probs = self.predict_proba([text])[0]
        label = self.predict([text])[0]
        return {
            "label": label,
            "confidence": float(np.max(probs)),
            "probabilities": {cls: float(p) for cls, p in zip(self._classes, probs)},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted classifier to *path* (pickle)."""
        self._require_fitted()
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "BullyingClassifier":
        """Deserialise a fitted classifier from *path*."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BullyingClassifier, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> list[str]:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Classifier has not been trained yet. Call fit() first.")
