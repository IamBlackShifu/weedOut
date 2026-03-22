"""
Feature extraction for cyberbullying detection.

Produces a combined feature vector from:
  - TF-IDF bag-of-words / n-grams
  - Sentiment score (positive, negative, neutral ratio)
  - Toxicity lexicon hit-counts
  - Structural features (caps ratio, exclamation count, mention count)
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from weedout.preprocessing.text_cleaner import TextCleaner

# ---------------------------------------------------------------------------
# Toxicity / bullying seed lexicon (generic, demographically neutral)
# ---------------------------------------------------------------------------
# This list is intentionally concise and generic; it should be extended with
# a domain-specific lexicon obtained from labelled data.
TOXICITY_LEXICON: frozenset[str] = frozenset(
    [
        "idiot", "stupid", "moron", "loser", "worthless", "pathetic",
        "ugly", "dumb", "freak", "die", "kill", "hate", "disgusting",
        "trash", "garbage", "scum", "shut up", "go away", "nobody likes you",
        "you suck", "horrible", "terrible", "awful", "useless", "creep",
        "weirdo", "lame", "coward",
    ]
)
# NOTE: Many terms above are context-dependent (e.g., "terrible weather" is not
# bullying). These lexicon hit-counts are used as weak signals alongside
# TF-IDF and structural features, not as stand-alone classifiers. Replace or
# supplement this list with a domain-specific lexicon derived from labelled data
# to improve precision.

_cleaner = TextCleaner(remove_stopwords=False, stem=False)


# ---------------------------------------------------------------------------
# Hand-crafted (structural) feature helpers
# ---------------------------------------------------------------------------

def _caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _exclamation_count(text: str) -> int:
    return text.count("!")


def _question_count(text: str) -> int:
    return text.count("?")


def _mention_count(text: str) -> int:
    return len(re.findall(r"@\w+", text))


def _toxicity_hit_count(text: str) -> int:
    """Number of toxicity-lexicon terms present in *text* (case-insensitive)."""
    lower = text.lower()
    return sum(1 for term in TOXICITY_LEXICON if term in lower)


def _structural_features(texts: Iterable[str]) -> np.ndarray:
    """Return an (n, 5) array of structural features for a batch of texts."""
    rows = []
    for text in texts:
        rows.append(
            [
                _caps_ratio(text),
                _exclamation_count(text),
                _question_count(text),
                _mention_count(text),
                _toxicity_hit_count(text),
            ]
        )
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Public feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Transform a list of raw post strings into a dense feature matrix.

    The extractor combines TF-IDF n-gram features with hand-crafted structural
    features via ``sklearn.pipeline.FeatureUnion``.

    Parameters
    ----------
    ngram_range : tuple[int, int]
        The lower and upper boundary of the range of n-gram sizes.
    max_features : int | None
        Maximum number of TF-IDF vocabulary terms. ``None`` means no limit.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 10_000,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], _y=None) -> "FeatureExtractor":
        cleaned = [_cleaner.clean(t) for t in texts]
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(cleaned)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("FeatureExtractor must be fitted before transform().")
        cleaned = [_cleaner.clean(t) for t in texts]
        result = self._pipeline.transform(cleaned)
        # Convert sparse matrix to dense array for uniform downstream handling
        if hasattr(result, "toarray"):
            return result.toarray()
        return np.asarray(result)

    def fit_transform(self, texts: list[str], _y=None) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=1,
        )
        structural = FunctionTransformer(_structural_features, validate=False)

        union = FeatureUnion(
            transformer_list=[
                ("tfidf", tfidf),
                ("structural", structural),
            ]
        )
        return Pipeline([("features", union)])
