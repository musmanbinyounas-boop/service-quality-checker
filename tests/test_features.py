"""Synthetic unit tests for sqc.features – no real data access."""

from __future__ import annotations

import pandas as pd
import pytest

from sqc import config
from sqc.features import clean, make_label


def _make_df(**kwargs) -> pd.DataFrame:
    defaults = {
        "Timestamp": "2020.01.01_00.00.00",
        "NetworkMode": "LTE",
        "RSRP": -90.0,
        "RSRQ": -10.0,
        "SNR": 10.0,
        "CQI": 8.0,
        "Speed": 50.0,
        config.THROUGHPUT_COL: 6000.0,
        "NRxRSRQ": 0.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_label_boundary_pass():
    df = make_label(clean(_make_df(**{config.THROUGHPUT_COL: 5000.0})))
    assert df[config.LABEL_COL].iloc[0] == 1


def test_label_boundary_fail():
    df = make_label(clean(_make_df(**{config.THROUGHPUT_COL: 4999.0})))
    assert df[config.LABEL_COL].iloc[0] == 0


def test_label_uses_default_threshold():
    df = make_label(clean(_make_df(**{config.THROUGHPUT_COL: float(config.PASS_THRESHOLD_KBPS)})))
    assert df[config.LABEL_COL].iloc[0] == 1


def test_rsrp_sentinel_scrubbed():
    df = clean(_make_df(RSRP=float(config.RSRP_SENTINEL)))
    assert pd.isna(df["RSRP"].iloc[0])


def test_rsrp_valid_not_scrubbed():
    df = clean(_make_df(RSRP=-90.0))
    assert df["RSRP"].iloc[0] == pytest.approx(-90.0)


def test_nrx_rsrq_sentinel_scrubbed():
    df = clean(_make_df(NRxRSRQ=float(config.NRX_RSRQ_GARBAGE)))
    assert pd.isna(df["NRxRSRQ"].iloc[0])


def test_timestamp_parsed():
    df = clean(_make_df(Timestamp="2020.06.15_12.30.00"))
    assert df["ts"].iloc[0] == pd.Timestamp("2020-06-15 12:30:00")


def test_numeric_coercion():
    df = clean(_make_df(RSRP="-90"))
    assert df["RSRP"].iloc[0] == pytest.approx(-90.0)
