"""Hermetic unit tests for sqc.drift – no real MongoDB required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from sqc import config
from sqc.drift import load_recent_from_mongo


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
