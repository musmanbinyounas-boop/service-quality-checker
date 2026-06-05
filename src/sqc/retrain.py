"""Stage 8 – Simulated Continuous Training pipeline.

CT IS SIMULATED: production traffic is unlabeled.  The 'new data arrivals'
are the most-recent 20% of the TRAINING dataset, held back temporally to
stand in for an arriving labeled batch.  Metrics are indicative only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sqc import config
from sqc.drift import load_recent_from_mongo, run_drift
from sqc.train import build_pipelines

_BANNER = """\
=================================================================
  SIMULATED CONTINUOUS TRAINING
  NOTE: Production traffic is unlabeled.  The "new data
  arrivals" are the most-recent 20%% of the TRAINING dataset,
  held back temporally to simulate an arriving labeled batch.
  All metrics are indicative only -- not an unbiased prod eval.
================================================================="""


# --------------------------------------------------------------------------- #
# Pure helpers (testable without disk I/O)                                    #
# --------------------------------------------------------------------------- #


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (D_old, D_new): earliest 80%% and latest 20%% of rows.

    Caller must ensure *df* is already sorted by ts before calling.
    """
    cutoff = int(len(df) * 0.8)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def promotion_gate(baseline_f1: float, retrained_f1: float) -> bool:
    """Return True iff retrained F1 meets or exceeds baseline F1."""
    return retrained_f1 >= baseline_f1


# --------------------------------------------------------------------------- #
# Internal training helpers                                                   #
# --------------------------------------------------------------------------- #


def _eval_pipeline(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }


def _best_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series):
    """Train all pipeline types; return (name, fitted_pipeline, metrics)."""
    pipelines = build_pipelines()
    for p in pipelines.values():
        p.fit(X_train, y_train)
    best_name = max(
        pipelines,
        key=lambda n: f1_score(
            y_test, pipelines[n].predict(X_test), pos_label=1, zero_division=0
        ),
    )
    metrics = _eval_pipeline(pipelines[best_name], X_test, y_test)
    return best_name, pipelines[best_name], metrics


def _load_dataset() -> pd.DataFrame:
    if not config.PROCESSED_FILE.exists():
        print("dataset.parquet not found — building from raw data ...")
        from sqc.features import build_dataset
        try:
            build_dataset()
        except Exception as exc:
            raise SystemExit(
                f"Could not build dataset: {exc}\n"
                "Run: python scripts/get_data.py && python -m sqc.features"
            ) from exc
    return pd.read_parquet(config.PROCESSED_FILE).sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main orchestrator                                                            #
# --------------------------------------------------------------------------- #


def run_ct(
    drift_source: str = "holdback",
    limit: int = 2000,
    force: bool = False,
    promote: bool = False,
    _df: pd.DataFrame | None = None,
) -> dict:
    """Execute the simulated CT pipeline; return a result dict.

    *_df* injects a pre-built DataFrame (tests only) — skips disk I/O.
    """
    print(_BANNER)

    df = _df if _df is not None else _load_dataset()
    d_old, d_new = temporal_split(df)

    print(f"\nDataset : {len(df):,} rows")
    print(f"  D_old (reference)  : {len(d_old):,} rows")
    print(f"  D_new (new arrivals): {len(d_new):,} rows")
    if "ts" in df.columns:
        print(f"  D_old range: {d_old['ts'].min()} — {d_old['ts'].max()}")
        print(f"  D_new range: {d_new['ts'].min()} — {d_new['ts'].max()}")

    # ------------------------------------------------------------------ #
    # [1/4] Drift detection                                                #
    # ------------------------------------------------------------------ #
    print(f"\n[1/4] Drift detection  (source={drift_source})")
    if drift_source == "mongo":
        drift_df = load_recent_from_mongo(limit=limit)
        if drift_df.empty:
            print("  WARNING: No Mongo data — falling back to holdback split.")
            drift_df = d_new[list(config.FEATURES)]
    else:
        drift_df = d_new[list(config.FEATURES)]

    drift_result = run_drift(drift_df)  # prints its own warning when n < MIN_DRIFT_SAMPLE
    is_insufficient = drift_result.get("status") == "insufficient_data"

    if is_insufficient:
        print(f"  INSUFFICIENT DATA: {drift_result['n']} rows "
              f"(need >= {drift_result['min_sample']}) — PSI suppressed, alert=False.")
    else:
        for feat, v in drift_result["features"].items():
            print(f"  {feat:8s}: PSI={v['psi']:.4f}  ({v['severity']})")
        print(f"  max_PSI={drift_result['max_psi']:.4f}  alert={drift_result['alert']}")

    should_retrain = drift_result["alert"] or force
    if not should_retrain:
        print("\nNo drift alert detected and --force not set — no retrain needed.")
        return {"drift": drift_result, "retrained": False, "reason": "no_drift_no_force"}

    reason = "forced" if force else "drift_alert"
    print(f"\n  Proceeding with retrain  (reason: {reason})")

    # ------------------------------------------------------------------ #
    # [2/4] Build training sets                                            #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Building training sets")
    X_old = d_old[list(config.FEATURES)]
    y_old = d_old[config.LABEL_COL]

    X_train_old, X_test_fixed, y_train_old, y_test_fixed = train_test_split(
        X_old, y_old, test_size=0.2, stratify=y_old, random_state=config.RANDOM_STATE
    )
    X_new = d_new[list(config.FEATURES)]
    y_new = d_new[config.LABEL_COL]
    X_retrain = pd.concat([X_train_old, X_new], ignore_index=True)
    y_retrain = pd.concat([y_train_old, y_new], ignore_index=True)

    print(f"  baseline  train={len(X_train_old):,}  |  test_fixed={len(X_test_fixed):,}")
    print(f"  retrained train={len(X_retrain):,}  |  (same test_fixed)")

    # ------------------------------------------------------------------ #
    # [3/4] Train baseline and retrained models                            #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Training models")
    base_name, base_model, base_m = _best_pipeline(
        X_train_old, y_train_old, X_test_fixed, y_test_fixed
    )
    ret_name, ret_model, ret_m = _best_pipeline(
        X_retrain, y_retrain, X_test_fixed, y_test_fixed
    )

    print(f"  Baseline  ({base_name:20s}): "
          f"P={base_m['precision']:.3f}  R={base_m['recall']:.3f}  F1={base_m['f1']:.3f}")
    print(f"  Retrained ({ret_name:20s}): "
          f"P={ret_m['precision']:.3f}  R={ret_m['recall']:.3f}  F1={ret_m['f1']:.3f}")

    # ------------------------------------------------------------------ #
    # [4/4] Promotion gate                                                 #
    # ------------------------------------------------------------------ #
    print("\n[4/4] Promotion gate")
    gate_passed = promotion_gate(base_m["f1"], ret_m["f1"])
    recommendation = "promote" if gate_passed else "keep_current"
    symbol = ">=" if gate_passed else "<"
    print(f"  retrained F1 ({ret_m['f1']:.3f}) {symbol} baseline F1 ({base_m['f1']:.3f})"
          f"  =>  RECOMMENDATION: {recommendation.upper()}")

    promoted = False
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if promote and gate_passed:
        bundle = {
            "model": ret_model,
            "model_type": ret_name,
            "features": list(config.FEATURES),
            "threshold_kbps": config.PASS_THRESHOLD_KBPS,
            "metrics": ret_m,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_train": int(len(X_retrain)),
        }
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = config.MODELS_DIR / "model.joblib"
        versioned = config.MODELS_DIR / f"model_{ts_str}.joblib"
        joblib.dump(bundle, model_path, compress=3)
        joblib.dump(bundle, versioned, compress=3)
        print(f"  PROMOTED -> {model_path}")
        print(f"  PROMOTED -> {versioned}")
        promoted = True
    elif promote and not gate_passed:
        print("  --promote passed but gate not met — production model unchanged.")
    else:
        print("  --promote not passed — candidate model NOT written to production path.")

    # Always save candidate model for artifact upload
    ct_dir = config.REPORTS_DIR / "ct"
    ct_dir.mkdir(parents=True, exist_ok=True)
    candidate_bundle = {
        "model": ret_model,
        "model_type": ret_name,
        "features": list(config.FEATURES),
        "threshold_kbps": config.PASS_THRESHOLD_KBPS,
        "metrics": ret_m,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_retrain)),
    }
    candidate_path = ct_dir / f"model_candidate_{ts_str}.joblib"
    joblib.dump(candidate_bundle, candidate_path, compress=3)

    # Write CT report
    report: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "drift_source": drift_source,
        "drift_status": drift_result.get("status", "ok"),
        "drift": drift_result,
        "reason": reason,
        "n_d_old": len(d_old),
        "n_d_new": len(d_new),
        "baseline": {"model_type": base_name, **base_m},
        "retrained": {"model_type": ret_name, **ret_m},
        "gate_passed": gate_passed,
        "promoted": promoted,
        "recommendation": recommendation,
        "candidate_model": str(candidate_path),
    }
    report_path = ct_dir / f"ct_{ts_str}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  CT report -> {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Continuous Training")
    parser.add_argument(
        "--drift-source", choices=["holdback", "mongo"], default="holdback",
        help="Data source for drift detection"
    )
    parser.add_argument("--limit", type=int, default=2000, help="Max Mongo docs for drift")
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even without a drift alert (always produces a CT report)"
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Overwrite models/model.joblib if the promotion gate passes"
    )
    args = parser.parse_args()
    run_ct(
        drift_source=args.drift_source,
        limit=args.limit,
        force=args.force,
        promote=args.promote,
    )
