"""Tests for the feature extraction pipeline."""

import numpy as np
import pytest

from weedout.features.feature_extractor import (
    FeatureExtractor,
    _caps_ratio,
    _exclamation_count,
    _mention_count,
    _toxicity_hit_count,
)


class TestStructuralFeatures:
    def test_caps_ratio_all_caps(self):
        assert _caps_ratio("HELLO") == 1.0

    def test_caps_ratio_no_caps(self):
        assert _caps_ratio("hello") == 0.0

    def test_caps_ratio_mixed(self):
        ratio = _caps_ratio("Hello")
        assert 0 < ratio < 1

    def test_caps_ratio_empty(self):
        assert _caps_ratio("") == 0.0

    def test_exclamation_count(self):
        assert _exclamation_count("Wow! Really! No!") == 3
        assert _exclamation_count("plain text") == 0

    def test_mention_count(self):
        assert _mention_count("@alice and @bob") == 2
        assert _mention_count("no mentions") == 0

    def test_toxicity_hit(self):
        assert _toxicity_hit_count("you are so stupid and ugly") >= 2
        assert _toxicity_hit_count("have a nice day") == 0


class TestFeatureExtractor:
    @pytest.fixture()
    def fitted_extractor(self):
        texts = [
            "I hate you so much",
            "You are so stupid",
            "Have a great day",
            "I love this",
        ]
        fe = FeatureExtractor(ngram_range=(1, 1), max_features=500)
        fe.fit(texts)
        return fe

    def test_fit_transform_shape(self):
        texts = ["hello world", "bad person", "great day", "horrible person"]
        fe = FeatureExtractor(ngram_range=(1, 1), max_features=100)
        X = fe.fit_transform(texts)
        assert X.shape[0] == 4  # four samples

    def test_transform_after_fit(self, fitted_extractor):
        X = fitted_extractor.transform(["test text"])
        assert X.shape[0] == 1

    def test_requires_fit_before_transform(self):
        fe = FeatureExtractor()
        with pytest.raises(RuntimeError):
            fe.transform(["some text"])

    def test_output_is_numeric(self, fitted_extractor):
        X = fitted_extractor.transform(["some text here"])
        assert np.isfinite(X).all()
