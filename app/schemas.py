"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from sqc import config

_B = config.FEATURE_BOUNDS


class PredictRequest(BaseModel):
    RSRP: float = Field(..., ge=_B["RSRP"][0], le=_B["RSRP"][1])
    RSRQ: float = Field(..., ge=_B["RSRQ"][0], le=_B["RSRQ"][1])
    SNR: Optional[float] = Field(None, ge=_B["SNR"][0], le=_B["SNR"][1])
    CQI: Optional[float] = Field(None, ge=_B["CQI"][0], le=_B["CQI"][1])
    Speed: float = Field(..., ge=_B["Speed"][0], le=_B["Speed"][1])


class PredictResponse(BaseModel):
    prediction: str
    label: int
    pass_probability: float
    threshold_kbps: int
    model_type: str
    latency_ms: float
