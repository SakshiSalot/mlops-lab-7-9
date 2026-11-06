"""API tests with FastAPI TestClient (B5: valid, missing field, wrong type, business rule, health)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.predict import predict
from tests.conftest import VALID_HOUSE


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path=model_path)) as c:      # runs startup -> loads model
        yield c


def without(field: str) -> dict:
    return {k: v for k, v in VALID_HOUSE.items() if k != field}


def test_health_reports_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_valid_request_returns_prediction(client, loaded):
    r = client.post("/predict", json=VALID_HOUSE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"prediction", "unit", "model_version", "request_id"}
    expected = float(predict(loaded, [{**VALID_HOUSE, "has_garage": 1}])[0])
    assert body["prediction"] == pytest.approx(expected, abs=1e-9)
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_missing_required_field_returns_422(client):
    r = client.post("/predict", json=without("bedrooms"))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert body["details"][0]["field"] == "bedrooms" and body["details"][0]["type"] == "missing"


@pytest.mark.parametrize("field,value", [("area_sqft", "1850"), ("bedrooms", 2.5), ("has_garage", "yes"),
                                         ("locality", 5)])
def test_wrong_data_type_returns_422(client, field, value):
    r = client.post("/predict", json={**VALID_HOUSE, field: value})
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == field


@pytest.mark.parametrize("field,value", [("area_sqft", 50), ("year_built", 2031), ("locality", "Uptown")])
def test_out_of_range_or_unknown_value_returns_422(client, field, value):
    r = client.post("/predict", json={**VALID_HOUSE, field: value})
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == field


def test_business_rule_violation_returns_controlled_422(client):
    r = client.post("/predict", json={**VALID_HOUSE, "bedrooms": 1, "bathrooms": 4})
    assert r.status_code == 422
    assert "cannot exceed bedrooms" in r.json()["details"][0]["message"]


def test_unknown_field_is_rejected(client):
    r = client.post("/predict", json={**VALID_HOUSE, "owner_name": "someone"})
    assert r.status_code == 422


def test_service_survives_invalid_requests(client):
    for bad in [{}, without("locality"), {**VALID_HOUSE, "area_sqft": "big"}]:
        client.post("/predict", json=bad)
    assert client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=VALID_HOUSE).status_code == 200


def test_internal_errors_do_not_leak_details(model_path, monkeypatch):
    app = create_app(model_path=model_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*args, **kwargs):
            raise RuntimeError("secret detail: /internal/path/model.joblib")
        monkeypatch.setattr(app.state.model.pipeline, "predict", boom)
        r = c.post("/predict", json=VALID_HOUSE)
    assert r.status_code == 500
    assert r.json()["error"] == "internal_error"
    assert "secret" not in r.text and "Traceback" not in r.text


def test_missing_model_gives_503_not_crash(tmp_path):
    with TestClient(create_app(model_path=str(tmp_path / "missing.joblib"))) as c:
        h = c.get("/health")
        p = c.post("/predict", json=VALID_HOUSE)
    assert h.status_code == 503 and h.json()["model_loaded"] is False
    assert p.status_code == 503 and p.json()["error"] == "model_unavailable"


def test_model_info_matches_loaded_artifact(client, loaded):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_sha256"] == loaded.metadata["model_sha256"]
