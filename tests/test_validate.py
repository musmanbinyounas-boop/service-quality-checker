"""Synthetic unit tests for sqc.validate – no real data access."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from sqc import config
from sqc.validate import validate


def _valid_row(**kwargs) -> pd.DataFrame:
    defaults = {
        "RSRP": -90.0,
        "RSRQ": -10.0,
        "SNR": 10.0,
        "CQI": 8.0,
        "Speed": 50.0,
        config.THROUGHPUT_COL: 6000.0,
        config.LABEL_COL: 1,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_schema_accepts_valid():
    validate(_valid_row())


def test_schema_allows_nan_radio_features():
    validate(_valid_row(RSRP=float("nan"), SNR=float("nan")))


def test_schema_rejects_rsrp_out_of_range():
    with pytest.raises(SchemaErrors):
        validate(_valid_row(RSRP=-200.0))


def test_schema_rejects_label_2():
    with pytest.raises(SchemaErrors):
        validate(_valid_row(**{config.LABEL_COL: 2}))


def test_schema_rejects_negative_throughput():
    with pytest.raises(SchemaErrors):
        validate(_valid_row(**{config.THROUGHPUT_COL: -1.0}))
