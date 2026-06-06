"""Stage 5 – Model training, evaluation, and artifact persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandera.errors import SchemaErrors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sqc import config
from sqc.validate import validate


def _load_data() -> pd.DataFrame:
    """Load and validate the processed dataset from disk.

    If dataset.parquet is missing it is built automatically by calling
    features.build_dataset().  Raises SystemExit if Pandera validation fails,
    ensuring training never runs on corrupt data.
    """
    if not config.PROCESSED_FILE.exists():
        print("dataset.parquet not found -- building it now ...")
        from sqc.features import build_dataset

        build_dataset()
    df = pd.read_parquet(config.PROCESSED_FILE)
    try:
        validate(df)
    except SchemaErrors as exc:
        raise SystemExit(
            f"Validation failed -- fix the data before training.\n{exc}"
        ) from exc
    return df


def build_pipelines() -> dict[str, Pipeline]:
    """Return a fresh dict of named, unfitted sklearn Pipelines.

    Each pipeline pairs a SimpleImputer (median strategy, handles missing radio
    measurements) with its classifier.  Returns both LogisticRegression and
    RandomForest variants so the caller can train and compare them fairly.
    Exposed as a public function so retrain.py can reuse the same model
    configuration without duplicating hyperparameters.
    """
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def _compute_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted pipeline on the held-out test set and return a metrics dict.

    Reports precision, recall, and F1 for the positive (Pass) class, plus the
    full classification report string and the confusion matrix as a nested list.
    All scores use pos_label=1 and zero_division=0 to handle edge cases cleanly.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def _feature_stats(X_train: pd.DataFrame) -> dict:
    """Per-feature reference distribution on the training set (for PSI drift, stage 7)."""
    stats = {}
    for feat in config.FEATURES:
        col = X_train[feat].dropna()
        edges = np.quantile(col.values, np.linspace(0, 1, 11))
        stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "quantile_10_bins": [float(v) for v in edges],
        }
    return stats


def train_from_frame(df: pd.DataFrame) -> dict:
    """Train both pipelines on *df*; return result dict with no disk I/O.

    Exposed as a public function so tests can pass a synthetic frame directly
    without touching the filesystem.
    """
    X = df[list(config.FEATURES)]
    y = df[config.LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    pipelines = build_pipelines()
    all_metrics: dict[str, dict] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        all_metrics[name] = _compute_metrics(pipeline, X_test, y_test)

    winner = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
    best_metrics = all_metrics[winner]

    bundle = {
        "model": pipelines[winner],
        "model_type": winner,
        "features": list(config.FEATURES),
        "threshold_kbps": config.PASS_THRESHOLD_KBPS,
        # classification_report is verbose text; keep bundle metrics numerical
        "metrics": {k: v for k, v in best_metrics.items() if k != "classification_report"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
    }

    return {
        "bundle": bundle,
        "all_metrics": {**all_metrics, "winner": winner},
        "feature_stats": _feature_stats(X_train),
    }


def _save_artifacts(result: dict) -> None:
    """Persist the training outputs to disk.

    Writes three artifacts: the model bundle (models/model.joblib and a
    timestamped versioned copy), the evaluation metrics for both pipelines
    (reports/metrics.json), and the per-feature reference distribution used
    by drift detection (reports/train_feature_stats.json).
    """
    bundle = result["bundle"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "model.joblib"
    versioned_path = config.MODELS_DIR / f"model_{ts}.joblib"
    joblib.dump(bundle, model_path, compress=3)
    joblib.dump(bundle, versioned_path, compress=3)

    metrics_path = config.REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(result["all_metrics"], fh, indent=2, default=str)

    stats_path = config.REPORTS_DIR / "train_feature_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(result["feature_stats"], fh, indent=2)

    print(f"Saved model     -> {model_path}")
    print(f"Saved versioned -> {versioned_path}")
    print(f"Saved metrics   -> {metrics_path}")
    print(f"Saved stats     -> {stats_path}")


def train() -> dict:
    """Load + validate data, train models, save all artifacts; return result."""
    df = _load_data()
    result = train_from_frame(df)
    _save_artifacts(result)
    return result


if __name__ == "__main__":
    result = train()
    am = result["all_metrics"]

    print()
    print("=" * 60)
    for name in ("LogisticRegression", "RandomForest"):
        m = am[name]
        print(f"  {name:20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\n  Winner: {am['winner']}")
    print("=" * 60)
    print()
    print("Classification report (winner):")
    print(am[am["winner"]]["classification_report"])
