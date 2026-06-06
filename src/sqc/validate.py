"""Stage 3 – Pandera schema validation."""

from __future__ import annotations

import pandas as pd

try:
    from pandera.pandas import Check, Column, DataFrameSchema
except ModuleNotFoundError:
    from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from sqc import config


def _build_schema() -> DataFrameSchema:
    """Build and return the Pandera schema used to validate the processed dataset.

    Constructs nullable float columns for each radio feature (RSRP, RSRQ, SNR,
    CQI, Speed) with empirically derived bounds, plus non-nullable columns for
    the throughput target and the binary label.  Called once at module load time.
    """
    feature_cols = {
        name: Column(float, Check.in_range(*config.FEATURE_BOUNDS[name]), nullable=True)
        for name in config.FEATURES
    }
    return DataFrameSchema(
        columns={
            **feature_cols,
            config.THROUGHPUT_COL: Column(float, Check.ge(0), nullable=False),
            config.LABEL_COL: Column(int, Check.isin([0, 1]), nullable=False),
        },
        coerce=True,
        strict=False,
    )


_SCHEMA = _build_schema()


def validate(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    """Validate *df* against the QoS schema; return validated frame."""
    return _SCHEMA.validate(df, lazy=lazy)


if __name__ == "__main__":
    df = pd.read_parquet(config.PROCESSED_FILE)
    try:
        validate(df)
        print("VALIDATION PASSED")
    except SchemaErrors as exc:
        print(f"VALIDATION FAILED – {len(exc.failure_cases)} failure(s)")
        print(exc.failure_cases.head(20))
