"""Hermetic tests for /dashboard and /dashboard/data — no real MongoDB."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _make_docs(n: int, pass_fraction: float = 0.5) -> list[dict]:
    docs = []
    for i in range(n):
        label = 1 if i < int(n * pass_fraction) else 0
        docs.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "pass_probability": 0.8 if label == 1 else 0.2,
            "latency_ms": 10.0 + i,
            "features": {
                "RSRP": -80.0, "RSRQ": -10.0, "SNR": 15.0, "CQI": 10.0, "Speed": 5.0,
            },
        })
    return docs


def _fake_col(docs: list[dict]) -> MagicMock:
    col = MagicMock()
    col.find.return_value.sort.return_value.limit.return_value = docs
    return col


# ---------------------------------------------------------------------------
# /dashboard/data — no Mongo
# ---------------------------------------------------------------------------

def test_data_no_mongo_returns_available_false(monkeypatch, client):
    monkeypatch.setattr("app.main.get_collection", lambda: None)
    resp = client.get("/dashboard/data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "reason" in data


def test_data_mongo_exception_returns_available_false(monkeypatch, client):
    def _boom():
        raise RuntimeError("unreachable host")
    monkeypatch.setattr("app.main.get_collection", _boom)
    resp = client.get("/dashboard/data")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


# ---------------------------------------------------------------------------
# /dashboard/data — empty collection
# ---------------------------------------------------------------------------

def test_data_empty_collection_returns_zero_total(monkeypatch, client):
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col([]))
    resp = client.get("/dashboard/data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["total_predictions"] == 0
    assert data["drift"]["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# /dashboard/data — few docs → totals correct, drift = insufficient_data
# ---------------------------------------------------------------------------

def test_data_few_docs_totals_correct(monkeypatch, client):
    docs = _make_docs(10, pass_fraction=0.5)
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col(docs))
    resp = client.get("/dashboard/data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["total_predictions"] == 10
    assert data["pass_count"] == 5
    assert data["fail_count"] == 5
    assert abs(data["pass_rate"] - 0.5) < 0.01


def test_data_few_docs_drift_insufficient(monkeypatch, client):
    docs = _make_docs(10, pass_fraction=0.5)
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col(docs))
    data = client.get("/dashboard/data").json()
    assert data["drift"]["status"] == "insufficient_data"
    assert data["drift"]["alert"] is False
    assert data["drift"]["n"] == 10


def test_data_75pct_pass_rate(monkeypatch, client):
    docs = _make_docs(20, pass_fraction=0.75)
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col(docs))
    data = client.get("/dashboard/data").json()
    assert data["pass_count"] == 15
    assert data["fail_count"] == 5
    assert abs(data["pass_rate"] - 0.75) < 0.01


def test_data_prob_histogram_has_ten_bins(monkeypatch, client):
    docs = _make_docs(10)
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col(docs))
    data = client.get("/dashboard/data").json()
    assert len(data["prob_histogram"]) == 10


def test_data_recent_capped_at_20(monkeypatch, client):
    docs = _make_docs(30)
    monkeypatch.setattr("app.main.get_collection", lambda: _fake_col(docs))
    data = client.get("/dashboard/data").json()
    assert len(data["recent"]) == 20


# ---------------------------------------------------------------------------
# /dashboard HTML + / redirect
# ---------------------------------------------------------------------------

def test_dashboard_returns_200_html(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Predict" in resp.text
    assert "Monitor" in resp.text
    assert "/predict" in resp.text       # fetch URL present
    assert "/dashboard/data" in resp.text


def test_root_redirects_and_returns_200(client):
    """GET / follows redirect to /dashboard and yields 200 HTML."""
    resp = client.get("/", follow_redirects=True)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
