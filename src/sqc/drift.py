"""Stage 7 – PSI feature drift detection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sqc import config
from sqc.store import get_collection

_PSI_MEDIUM = 0.1
_PSI_HIGH = 0.2


def _psi_one_feature(series: pd.Series, ref_bins: list[float]) -> float:
    """PSI of *series* against reference quantile-bin edges.

    The training-set bins are equal-mass (each holds 10% of reference rows),
    so the reference distribution is uniform across n_bins = len(edges)-1 bins.
    """
    edges = np.unique(np.asarray(ref_bins, dtype=float))
    if len(edges) < 3:  # need at least 2 bins
        return 0.0
    n_bins = len(edges) - 1
    ref_pct = np.ones(n_bins) / n_bins  # uniform by construction
    values = series.dropna().values
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    eps = 1e-6
    act_pct = np.clip(counts / total, eps, 1.0)
    ref_pct_c = np.clip(ref_pct, eps, 1.0)
    return float(np.sum((act_pct - ref_pct_c) * np.log(act_pct / ref_pct_c)))


def _severity(psi: float) -> str:
    if psi < _PSI_MEDIUM:
        return "low"
    if psi < _PSI_HIGH:
        return "medium"
    return "high"


def run_drift(df: pd.DataFrame, stats_path: Path | None = None) -> dict:
    """Compute per-feature PSI drift for *df* vs. the committed reference stats.

    Returns::

        {
            "features": {name: {"psi": float, "severity": str}, ...},
            "max_psi":  float,
            "alert":    bool,   # True when max_psi >= 0.2
        }
    """
    if stats_path is None:
        stats_path = config.REPORTS_DIR / "train_feature_stats.json"
    ref = json.loads(Path(stats_path).read_text())

    results: dict[str, dict] = {}
    for feat in config.FEATURES:
        col = df[feat] if feat in df.columns else pd.Series([], dtype=float)
        psi = _psi_one_feature(col, ref[feat]["quantile_10_bins"])
        results[feat] = {"psi": round(psi, 6), "severity": _severity(psi)}

    max_psi = max(v["psi"] for v in results.values())
    return {
        "features": results,
        "max_psi": round(max_psi, 6),
        "alert": max_psi >= _PSI_HIGH,
    }


def load_recent_from_mongo(limit: int = 2000) -> pd.DataFrame:
    """Pull the most recent *limit* prediction docs from MongoDB.

    Extracts the ``features`` sub-dict into a DataFrame with config.FEATURES
    columns.  Returns an empty DataFrame (with correct columns) when the
    collection is unavailable, empty, or an error occurs.
    """
    col = get_collection()
    if col is None:
        return pd.DataFrame(columns=list(config.FEATURES))
    try:
        docs = list(col.find({}, {"features": 1, "_id": 0}).sort("_id", -1).limit(limit))
    except Exception as exc:
        print(f"[drift] Mongo query failed: {exc}")
        return pd.DataFrame(columns=list(config.FEATURES))
    if not docs:
        return pd.DataFrame(columns=list(config.FEATURES))
    rows = [doc.get("features", {}) for doc in docs]
    return pd.DataFrame(rows, columns=list(config.FEATURES))


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="PSI feature drift detection")
    parser.add_argument(
        "--source", choices=["file", "mongo"], required=True,
        help="Data source: 'file' for CSV/parquet, 'mongo' for live predictions"
    )
    parser.add_argument("path", nargs="?", help="CSV or parquet path (--source file)")
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Max docs to pull from MongoDB (--source mongo)"
    )
    args = parser.parse_args()

    if args.source == "file":
        if not args.path:
            parser.error("--source file requires a path argument")
        p = Path(args.path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    else:
        df = load_recent_from_mongo(limit=args.limit)
        if df.empty:
            print("No data in MongoDB — cannot compute drift.")
            return

    n = len(df)
    if n < 100:
        print(f"WARNING: only {n} rows — drift estimate is unreliable (small sample).")

    result = run_drift(df)

    print("\nPSI Drift Report")
    print("=" * 52)
    for feat, v in result["features"].items():
        print(f"  {feat:8s}: PSI={v['psi']:.4f}  ({v['severity']})")
    print(f"\n  max_PSI : {result['max_psi']:.4f}")
    print(f"  alert   : {result['alert']}")

    out_dir = config.REPORTS_DIR / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"drift_{ts}.json"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "n_rows": n,
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Report -> {out_path}")


if __name__ == "__main__":
    _run_cli()
