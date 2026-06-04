"""Stage 6 – FastAPI prediction service.

Stateless: no log files written (logging/drift is stage 7).
Bundle is loaded once at startup and cached in module-level _bundle.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse
from sqc import config

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
def predict(req: PredictRequest) -> PredictResponse:
    features: list[str] = _bundle["features"]
    data = req.model_dump()
    # Map None -> np.nan so the pipeline's SimpleImputer handles missing values
    row = {f: (np.nan if data.get(f) is None else data[f]) for f in features}
    X = pd.DataFrame([row], columns=features)

    t0 = time.perf_counter()
    label = int(_bundle["model"].predict(X)[0])
    proba = float(_bundle["model"].predict_proba(X)[0][1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        prediction="Pass" if label == 1 else "Fail",
        label=label,
        pass_probability=round(proba, 4),
        threshold_kbps=int(_bundle["threshold_kbps"]),
        model_type=_bundle["model_type"],
        latency_ms=latency_ms,
    )
