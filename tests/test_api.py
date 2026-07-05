"""API tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import resolve_path

pytestmark = pytest.mark.skipif(
    not resolve_path("models/model.joblib").exists(),
    reason="Model artifacts not found — run `make train` first.",
)

client = TestClient(app)

VALID_PAYLOAD = {
    "temp": 0.5, "atemp": 0.48, "hum": 0.6, "windspeed": 0.2,
    "lag_1": 120, "lag_24": 140, "lag_168": 135,
    "roll_mean_24": 95.0, "roll_mean_168": 110.0,
    "hour_sin": 0.965, "hour_cos": -0.258,
    "month_sin": 0.5, "month_cos": 0.866,
    "weekday_sin": 0.781, "weekday_cos": 0.623,
    "season": 2, "weathersit": 1, "holiday": 0, "workingday": 1,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "bikeshare-demand-forecasting-api"


def test_predict_endpoint_valid_request():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_count"] >= 0


def test_predict_endpoint_missing_field():
    payload = dict(VALID_PAYLOAD)
    del payload["temp"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_season():
    payload = dict(VALID_PAYLOAD)
    payload["season"] = 9  # invalid
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_out_of_range_temp():
    payload = dict(VALID_PAYLOAD)
    payload["temp"] = 5.0  # must be normalized in [0, 1]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
