"""Hermetic unit tests for sqc.drift – no real MongoDB required."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from sqc import config
from sqc.drift import MIN_DRIFT_SAMPLE, load_recent_from_mongo, run_drift

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n: int, mode: str = "uniform") -> pd.DataFrame:
    """Build a synthetic feature DataFrame.

    mode="uniform"      — random values within FEATURE_BOUNDS (sufficient sample)
    mode="concentrated" — all rows at the same mid-range value (one bin gets all
                          mass → PSI explodes, guarantees alert=True)
    """
    rng = np.random.default_rng(7)
    if mode == "uniform":
        return pd.DataFrame({
            feat: rng.uniform(*config.FEATURE_BOUNDS[feat], n)
            for feat in config.FEATURES
        })
    # concentrated: mid-point of each feature's valid range
    return pd.DataFrame({
        feat: [(config.FEATURE_BOUNDS[feat][0] + config.FEATURE_BOUNDS[feat][1]) / 2] * n
        for feat in config.FEATURES
    })


def test_load_returns_empty_dataframe_when_no_collection(monkeypatch):
    """get_collection() -> None must yield empty DataFrame with FEATURES columns."""
    monkeypatch.setattr("sqc.drift.get_collection", lambda: None)
    df = load_recent_from_mongo(limit=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == list(config.FEATURES)


def test_load_returns_empty_dataframe_on_query_exception(monkeypatch):
    """Mongo query raising must yield empty DataFrame without propagating."""
    fake_col = MagicMock()
    fake_col.find.side_effect = Exception("network error")
    monkeypatch.setattr("sqc.drift.get_collection", lambda: fake_col)
    df = load_recent_from_mongo(limit=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == list(config.FEATURES)


def test_load_returns_empty_dataframe_when_collection_empty(monkeypatch):
    """Empty collection (no docs) must yield empty DataFrame."""
    fake_col = MagicMock()
    fake_col.find.return_value.sort.return_value.limit.return_value = []
    monkeypatch.setattr("sqc.drift.get_collection", lambda: fake_col)
    df = load_recent_from_mongo(limit=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# run_drift — small-sample guard
# ---------------------------------------------------------------------------

def test_run_drift_small_sample_returns_insufficient_data():
    """< MIN_DRIFT_SAMPLE rows must return status='insufficient_data', not inflated PSI."""
    df = _make_feature_df(n=MIN_DRIFT_SAMPLE - 1)
    result = run_drift(df)
    assert result["status"] == "insufficient_data"
    assert result["alert"] is False
    assert result["max_psi"] is None
    assert result["n"] == MIN_DRIFT_SAMPLE - 1
    for feat in config.FEATURES:
        assert result["features"][feat]["psi"] is None
        assert result["features"][feat]["severity"] == "unknown"


def test_run_drift_sufficient_data_returns_float_psi():
    """>= MIN_DRIFT_SAMPLE rows must return float PSI values, not None."""
    df = _make_feature_df(n=MIN_DRIFT_SAMPLE + 50)
    result = run_drift(df)
    assert result.get("status") != "insufficient_data"
    assert isinstance(result["max_psi"], float)
    assert isinstance(result["alert"], bool)
    for feat in config.FEATURES:
        assert isinstance(result["features"][feat]["psi"], float)
        assert result["features"][feat]["severity"] in ("low", "medium", "high")


def test_run_drift_concentrated_data_raises_alert():
    """Concentrated distribution (one bin gets all mass) with sufficient rows triggers alert."""
    df = _make_feature_df(n=MIN_DRIFT_SAMPLE + 50, mode="concentrated")
    result = run_drift(df)
    # All rows fall in one bin → other 9 bins are empty → PSI >> 0.2
    assert result.get("status") != "insufficient_data"
    assert result["alert"] is True
    assert result["max_psi"] >= 0.2
