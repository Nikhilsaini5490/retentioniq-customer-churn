import sys
from pathlib import Path

import asyncio

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retentioniq import api
from retentioniq.data import generate_demo_data
from retentioniq.service import train_and_save


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    artifact_path = tmp_path_factory.mktemp("api") / "model.joblib"
    train_and_save(generate_demo_data(400), artifact_path)
    original_model_path = api._model_path
    api._model_path = lambda: artifact_path
    api._load_configured_model.cache_clear()
    try:
        yield
    finally:
        api._load_configured_model.cache_clear()
        api._model_path = original_model_path


def request(method: str, path: str, payload: dict | None = None):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def customer(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "tenure_months": 8,
        "monthly_charges": 79.5,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_tickets_90d": 3,
        "payment_method": "Electronic check",
        "paperless_billing": True,
        "autopay": False,
        "senior_citizen": 0,
    }


def test_health_and_single_prediction(api_client):
    health = request("GET", "/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True

    response = request("POST", "/predict", customer("C-1"))
    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert prediction["customer_id"] == "C-1"
    assert 0 <= prediction["churn_probability"] <= 1


def test_batch_prediction_and_request_validation(api_client):
    response = request(
        "POST",
        "/predict/batch",
        {"customers": [customer("C-1"), customer("C-2")], "high_risk_threshold": 0.7},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2

    invalid = customer("C-3")
    invalid["contract"] = "Lifetime"
    assert request("POST", "/predict", invalid).status_code == 422