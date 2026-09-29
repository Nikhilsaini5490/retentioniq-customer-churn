"""FastAPI inference service for trained RetentionIQ artifacts."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from retentioniq.service import DEFAULT_MODEL_PATH, load_model_artifact, predict_customers

app = FastAPI(
    title="RetentionIQ API",
    version="1.0.0",
    description="Validated churn-risk inference for customer-retention workflows.",
)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(default=None, max_length=128)
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=10000)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    support_tickets_90d: int = Field(ge=0, le=100)
    payment_method: Literal["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
    paperless_billing: bool
    autopay: bool
    senior_citizen: int = Field(default=0, ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)
    high_risk_threshold: float = Field(default=0.65, ge=0.05, le=0.95)


def _model_path() -> Path:
    configured = os.getenv("RETENTIONIQ_MODEL_PATH")
    return Path(configured) if configured else DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def _load_configured_model() -> dict:
    return load_model_artifact(_model_path())


@app.get("/health")
def health() -> dict:
    try:
        artifact = _load_configured_model()
        return {"status": "healthy", "model_ready": True, "model": artifact["model_name"]}
    except (FileNotFoundError, ValueError, KeyError):
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "model_ready": False},
        )


def _score(records: list[CustomerFeatures], threshold: float) -> dict:
    try:
        artifact = _load_configured_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not trained. Run scripts/train_model.py first.") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Model artifact is invalid. Retrain the model.") from exc

    frame = pd.DataFrame([record.model_dump() for record in records])
    try:
        scored = predict_customers(artifact["pipeline"], frame, threshold)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model": artifact["model_name"],
        "threshold": threshold,
        "predictions": scored.to_dict(orient="records"),
    }


@app.post("/predict")
def predict(customer: CustomerFeatures, high_risk_threshold: float = 0.65) -> dict:
    if not 0.05 <= high_risk_threshold <= 0.95:
        raise HTTPException(status_code=422, detail="high_risk_threshold must be between 0.05 and 0.95.")
    return _score([customer], high_risk_threshold)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest) -> dict:
    return _score(request.customers, request.high_risk_threshold)