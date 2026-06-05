"""Stage 6/7 – FastAPI prediction service with MongoDB monitoring.

Stateless inference; Mongo write happens in a BackgroundTask so the DB
never adds latency to /predict and a down DB never breaks inference.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI

from app.schemas import PredictRequest, PredictResponse
from sqc import config
from sqc.store import build_prediction_record, save_prediction

_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = joblib.load(config.MODELS_DIR / "model.joblib")
    _bundle.update(loaded)
    yield


app = FastAPI(title="Service Quality Checker", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_type": _bundle["model_type"],
        "features": _bundle["features"],
        "trained_at": _bundle["trained_at"],
        "sklearn_version": _bundle["sklearn_version"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    features: list[str] = _bundle["features"]
    data = req.model_dump()
    row = {f: (np.nan if data.get(f) is None else data[f]) for f in features}
    X = pd.DataFrame([row], columns=features)

    t0 = time.perf_counter()
    label = int(_bundle["model"].predict(X)[0])
    proba = float(_bundle["model"].predict_proba(X)[0][1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    print(
        f'[predict] {{"label": {label}, "pass_probability": {proba:.4f}, '
        f'"model_type": "{_bundle["model_type"]}", "latency_ms": {latency_ms}}}'
    )

    features_for_record = {
        f: (None if pd.isna(v) else float(v)) for f, v in row.items()
    }
    record = build_prediction_record(
        features=features_for_record,
        label=label,
        pass_probability=proba,
        model_type=_bundle["model_type"],
        latency_ms=latency_ms,
    )
    background_tasks.add_task(save_prediction, record)

    return PredictResponse(
        prediction="Pass" if label == 1 else "Fail",
        label=label,
        pass_probability=round(proba, 4),
        threshold_kbps=int(_bundle["threshold_kbps"]),
        model_type=_bundle["model_type"],
        latency_ms=latency_ms,
    )
