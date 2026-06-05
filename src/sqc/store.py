"""Stage 7 – MongoDB persistence for prediction monitoring."""

from __future__ import annotations

import datetime

from sqc import config

_collection = None  # lazy singleton; None until first successful get_collection()


def get_collection():
    """Return the predictions Collection, or None if MongoDB is unavailable.

    Creates the MongoClient once (module-level singleton) with a short
    serverSelectionTimeoutMS so a down DB never blocks inference startup.
    Returns None gracefully when MONGODB_URI is unset (local / test envs).
    """
    global _collection
    if _collection is not None:
        return _collection
    if not config.MONGODB_URI:
        return None
    try:
        from pymongo import MongoClient

        client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)
        _collection = client[config.MONGO_DB_NAME][config.MONGO_COLLECTION]
        return _collection
    except Exception as exc:
        print(f"[store] MongoDB client creation failed: {exc}")
        return None


def build_prediction_record(
    features: dict,
    label: int,
    pass_probability: float,
    model_type: str,
    latency_ms: float,
) -> dict:
    """Build a Mongo document dict.  Pure function — no side effects."""
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "features": features,
        "label": int(label),
        "pass_probability": float(pass_probability),
        "model_type": str(model_type),
        "latency_ms": float(latency_ms),
    }


def save_prediction(record: dict) -> bool:
    """Insert *record* into the predictions collection.

    Returns True on success, False on any failure.  NEVER raises — monitoring
    must never break inference.
    """
    try:
        col = get_collection()
        if col is None:
            return False
        col.insert_one(record)
        return True
    except Exception as exc:
        print(f"[store] save_prediction failed: {exc}")
        return False
