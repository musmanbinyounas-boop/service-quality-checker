"""Smoke tests for sqc.train – synthetic data only, no disk I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sqc import config
from sqc.train import train_from_frame

_EXPECTED_BUNDLE_KEYS = {
    "model",
    "model_type",
    "features",
    "threshold_kbps",
    "metrics",
    "trained_at",
    "sklearn_version",
    "n_train",
}

_EXPECTED_METRIC_KEYS = {"precision", "recall", "f1", "confusion_matrix"}


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    """Build a balanced labelled DataFrame with valid feature ranges."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        feat: rng.uniform(*config.FEATURE_BOUNDS[feat], n)
        for feat in config.FEATURES
    })
    df[config.THROUGHPUT_COL] = rng.uniform(1000, 15000, n)
    # Exactly 50/50 so stratified split is always valid
    df[config.LABEL_COL] = [0, 1] * (n // 2)
    return df


@pytest.fixture(scope="module")
def train_result():
    return train_from_frame(_make_synthetic_df())


def test_bundle_keys(train_result):
    assert set(train_result["bundle"].keys()) == _EXPECTED_BUNDLE_KEYS


def test_bundle_metric_keys(train_result):
    assert set(train_result["bundle"]["metrics"].keys()) == _EXPECTED_METRIC_KEYS


def test_model_type_valid(train_result):
    assert train_result["bundle"]["model_type"] in ("LogisticRegression", "RandomForest")


def test_features_match_config(train_result):
    assert train_result["bundle"]["features"] == list(config.FEATURES)


def test_threshold_matches_config(train_result):
    assert train_result["bundle"]["threshold_kbps"] == config.PASS_THRESHOLD_KBPS


def test_model_predicts_single_row(train_result):
    model = train_result["bundle"]["model"]
    X = pd.DataFrame([{
        "RSRP": -90.0,
        "RSRQ": -10.0,
        "SNR": 10.0,
        "CQI": 8.0,
        "Speed": 30.0,
    }])
    pred = model.predict(X)
    assert pred[0] in (0, 1)


def test_model_predicts_with_nan_features(train_result):
    """Imputer must handle NaN features gracefully."""
    model = train_result["bundle"]["model"]
    X = pd.DataFrame([{
        "RSRP": float("nan"),
        "RSRQ": float("nan"),
        "SNR": 10.0,
        "CQI": 8.0,
        "Speed": 30.0,
    }])
    pred = model.predict(X)
    assert pred[0] in (0, 1)


def test_all_metrics_present(train_result):
    am = train_result["all_metrics"]
    assert "winner" in am
    for name in ("LogisticRegression", "RandomForest"):
        assert name in am
        assert "f1" in am[name]


def test_feature_stats_structure(train_result):
    stats = train_result["feature_stats"]
    assert set(stats.keys()) == set(config.FEATURES)
    for feat in config.FEATURES:
        s = stats[feat]
        assert set(s.keys()) == {"mean", "std", "min", "max", "quantile_10_bins"}
        assert len(s["quantile_10_bins"]) == 11
        # all values must be plain Python floats (JSON-serializable)
        assert isinstance(s["mean"], float)
        assert all(isinstance(v, float) for v in s["quantile_10_bins"])
