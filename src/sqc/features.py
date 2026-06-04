"""Stage 3 – Feature engineering and dataset builder."""

from __future__ import annotations

import pandas as pd

from sqc import config
from sqc.ingest import find_csv_files

_TIMESTAMP_FMT = "%Y.%m.%d_%H.%M.%S"
_KEEP_COLS = (
    ["ts", "Timestamp", "NetworkMode"]
    + list(config.FEATURES)
    + [config.THROUGHPUT_COL, config.LABEL_COL]
)


def load_raw() -> pd.DataFrame:
    """Read all raw CSVs and concatenate into a single frame."""
    frames = [
        pd.read_csv(p, na_values=config.NA_VALUES, low_memory=False)
        for p in find_csv_files()
    ]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerics, scrub sentinels, parse Timestamp → ts."""
    df = df.copy()
    for col in config.RAW_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "RSRP" in df.columns:
        df["RSRP"] = df["RSRP"].where(df["RSRP"] != config.RSRP_SENTINEL)
    if "NRxRSRQ" in df.columns:
        nrx = pd.to_numeric(df["NRxRSRQ"], errors="coerce")
        df["NRxRSRQ"] = nrx.where(nrx != config.NRX_RSRQ_GARBAGE)
    df["ts"] = pd.to_datetime(df["Timestamp"], format=_TIMESTAMP_FMT, errors="coerce")
    return df


def make_label(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Add binary pass/fail label column (1 = Pass, 0 = Fail)."""
    if threshold is None:
        threshold = config.PASS_THRESHOLD_KBPS
    df = df.copy()
    df[config.LABEL_COL] = (df[config.THROUGHPUT_COL] >= threshold).astype(int)
    return df


def build_dataset(threshold: float | None = None, save: bool = True) -> pd.DataFrame:
    """Load → clean → label → write dataset.parquet; return the frame."""
    df = load_raw()
    df = clean(df)
    df = make_label(df, threshold=threshold)
    df = df.dropna(subset=[config.THROUGHPUT_COL])
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep].sort_values("ts").reset_index(drop=True)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_FILE, index=False)
        print(f"Saved {len(df):,} rows -> {config.PROCESSED_FILE}")
    return df


if __name__ == "__main__":
    df = build_dataset()
    pass_rate = df[config.LABEL_COL].mean()
    print(f"Rows: {len(df):,}  Pass rate: {pass_rate:.1%}")
