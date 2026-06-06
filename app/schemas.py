"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from sqc import config

_B = config.FEATURE_BOUNDS


class PredictRequest(BaseModel):
    """Validated input schema for POST /predict.

    RSRP, RSRQ, and Speed are required.  SNR and CQI are optional (None is
    treated as missing and imputed by the pipeline's SimpleImputer).  All
    bounds are enforced by Pydantic using the empirical FEATURE_BOUNDS from
    config.py so out-of-range values are rejected before reaching the model.
    """

    RSRP: float = Field(..., ge=_B["RSRP"][0], le=_B["RSRP"][1])
    RSRQ: float = Field(..., ge=_B["RSRQ"][0], le=_B["RSRQ"][1])
    SNR: Optional[float] = Field(None, ge=_B["SNR"][0], le=_B["SNR"][1])
    CQI: Optional[float] = Field(None, ge=_B["CQI"][0], le=_B["CQI"][1])
    Speed: float = Field(..., ge=_B["Speed"][0], le=_B["Speed"][1])


class PredictResponse(BaseModel):
    """Output schema returned by POST /predict.

    'prediction' is the human-readable verdict ('Pass' or 'Fail').
    'label' is the integer class (1 = Pass, 0 = Fail).
    'pass_probability' is the model's estimated probability of Pass (0–1).
    'threshold_kbps' is the DL_bitrate threshold used to define Pass (5000).
    'latency_ms' is the end-to-end model inference time in milliseconds.
    """

    prediction: str
    label: int
    pass_probability: float
    threshold_kbps: int
    model_type: str
    latency_ms: float
