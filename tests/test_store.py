"""Hermetic unit tests for sqc.store – no real MongoDB required."""

from __future__ import annotations

from unittest.mock import MagicMock

from sqc.store import build_prediction_record, save_prediction


def test_build_record_has_expected_keys():
    record = build_prediction_record(
        features={"RSRP": -90.0, "RSRQ": -10.0, "SNR": 10.0, "CQI": 8.0, "Speed": 50.0},
        label=1,
        pass_probability=0.85,
        model_type="RandomForest",
        latency_ms=12.5,
    )
    assert set(record.keys()) == {
        "ts", "features", "label", "pass_probability", "model_type", "latency_ms"
    }


def test_build_record_types():
    record = build_prediction_record(
        features={"RSRP": -90.0},
        label=1,
        pass_probability=0.85,
        model_type="RandomForest",
        latency_ms=12.5,
    )
    assert isinstance(record["ts"], str)
    assert isinstance(record["label"], int)
    assert isinstance(record["pass_probability"], float)
    assert isinstance(record["latency_ms"], float)
    assert isinstance(record["features"], dict)


def test_build_record_ts_is_iso_string():
    record = build_prediction_record(
        features={}, label=0, pass_probability=0.1, model_type="LogReg", latency_ms=5.0
    )
    # Should parse as ISO timestamp without error
    from datetime import datetime
    datetime.fromisoformat(record["ts"])


def test_save_prediction_returns_false_when_collection_none(monkeypatch):
    monkeypatch.setattr("sqc.store.get_collection", lambda: None)
    assert save_prediction({"test": True}) is False


def test_save_prediction_calls_insert_one(monkeypatch):
    fake_col = MagicMock()
    monkeypatch.setattr("sqc.store.get_collection", lambda: fake_col)
    record = {"ts": "2024-01-01T00:00:00Z", "label": 1}
    result = save_prediction(record)
    assert result is True
    fake_col.insert_one.assert_called_once_with(record)


def test_save_prediction_returns_false_on_exception(monkeypatch):
    fake_col = MagicMock()
    fake_col.insert_one.side_effect = Exception("connection timeout")
    monkeypatch.setattr("sqc.store.get_collection", lambda: fake_col)
    assert save_prediction({"test": True}) is False
