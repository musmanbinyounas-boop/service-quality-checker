"""Hermetic unit tests for sqc.retrain – no real dataset, no real MongoDB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sqc import config
from sqc.retrain import promotion_gate, temporal_split
from sqc.train import build_pipelines

_N = 200


def _make_ts_df(n: int = _N) -> pd.DataFrame:
    """Synthetic ts-ordered, labeled DataFrame for CT tests."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        feat: rng.uniform(*config.FEATURE_BOUNDS[feat], n)
        for feat in config.FEATURES
    })
    df[config.THROUGHPUT_COL] = rng.uniform(1000, 15000, n)
    df[config.LABEL_COL] = [0, 1] * (n // 2)
    df["ts"] = timestamps
    return df.sort_values("ts").reset_index(drop=True)


@pytest.fixture(scope="module")
def split_result():
    return temporal_split(_make_ts_df())


# --------------------------------------------------------------------------- #
# Temporal split                                                               #
# --------------------------------------------------------------------------- #


def test_split_d_old_size(split_result):
    d_old, _ = split_result
    assert len(d_old) == int(_N * 0.8)


def test_split_d_new_size(split_result):
    _, d_new = split_result
    assert len(d_new) == _N - int(_N * 0.8)


def test_split_temporal_ordering(split_result):
    d_old, d_new = split_result
    assert d_old["ts"].max() <= d_new["ts"].min()


# --------------------------------------------------------------------------- #
# Training sanity                                                              #
# --------------------------------------------------------------------------- #


def test_baseline_trains_and_predicts():
    df = _make_ts_df()
    d_old, _ = temporal_split(df)
    X = d_old[list(config.FEATURES)]
    y = d_old[config.LABEL_COL]
    model = list(build_pipelines().values())[0]
    model.fit(X, y)
    preds = model.predict(X.iloc[:5])
    assert set(preds).issubset({0, 1})


def test_retrained_trains_and_predicts():
    df = _make_ts_df()
    d_old, d_new = temporal_split(df)
    X = pd.concat([d_old[list(config.FEATURES)], d_new[list(config.FEATURES)]])
    y = pd.concat([d_old[config.LABEL_COL], d_new[config.LABEL_COL]])
    model = list(build_pipelines().values())[0]
    model.fit(X, y)
    preds = model.predict(X.iloc[:5])
    assert set(preds).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Promotion gate                                                               #
# --------------------------------------------------------------------------- #


def test_gate_promotes_when_retrained_is_better():
    assert promotion_gate(baseline_f1=0.80, retrained_f1=0.85) is True


def test_gate_promotes_when_equal():
    assert promotion_gate(baseline_f1=0.80, retrained_f1=0.80) is True


def test_gate_rejects_when_retrained_is_worse():
    assert promotion_gate(baseline_f1=0.80, retrained_f1=0.75) is False
