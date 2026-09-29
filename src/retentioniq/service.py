"""Model artifact management and customer-scoring services."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from retentioniq.data import FEATURE_COLUMNS
from retentioniq.model import train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"


def train_and_save(data: pd.DataFrame, artifact_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    result = train_model(data)
    artifact = {
        "schema_version": 1,
        "model_name": result["model_name"],
        "pipeline": result["pipeline"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(data),
        "feature_columns": FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
    }
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return artifact


def load_model_artifact(artifact_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a trusted local artifact created by this project."""
    artifact = joblib.load(Path(artifact_path))
    if artifact.get("schema_version") != 1 or not hasattr(artifact.get("pipeline"), "predict_proba"):
        raise ValueError("Unsupported or invalid model artifact.")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match this application version.")
    return artifact


def predict_customers(
    pipeline: Any,
    customers: pd.DataFrame,
    high_risk_threshold: float = 0.65,
) -> pd.DataFrame:
    """Score new customers and assign an operational review band."""
    if not 0.05 <= high_risk_threshold <= 0.95:
        raise ValueError("high_risk_threshold must be between 0.05 and 0.95.")

    missing = sorted(set(FEATURE_COLUMNS) - set(customers.columns))
    if missing:
        raise ValueError(f"Customer data is missing required columns: {', '.join(missing)}")
    scoring_data = customers.copy()
    if "customer_id" in scoring_data:
        supplied_ids = scoring_data["customer_id"].astype("string").str.strip()
        non_empty_ids = supplied_ids[supplied_ids.notna() & supplied_ids.ne("")]
        if non_empty_ids.duplicated().any():
            raise ValueError("Non-empty customer_id values must be unique in a scoring file.")
        used_ids = set(non_empty_ids.tolist())
        normalized_ids = []
        for position, customer_id in enumerate(supplied_ids.tolist(), start=1):
            if pd.isna(customer_id) or not customer_id:
                candidate = f"customer-{position:05d}"
                suffix = 1
                while candidate in used_ids:
                    candidate = f"customer-{position:05d}-{suffix}"
                    suffix += 1
                used_ids.add(candidate)
                normalized_ids.append(candidate)
            else:
                normalized_ids.append(str(customer_id))
        scoring_data["customer_id"] = normalized_ids
    else:
        scoring_data.insert(
            0,
            "customer_id",
            [f"customer-{position:05d}" for position in range(1, len(scoring_data) + 1)],
        )

    features = scoring_data[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column in {"contract", "internet_service", "payment_method"}:
            features[column] = features[column].where(features[column].isna(), features[column].astype(str))
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    probabilities = pipeline.predict_proba(features)[:, 1]
    review_threshold = max(0.10, high_risk_threshold * 0.55)
    bands = [
        "High" if score >= high_risk_threshold else "Watch" if score >= review_threshold else "Low"
        for score in probabilities
    ]
    result = pd.DataFrame(
        {
            "customer_id": scoring_data["customer_id"].astype(str).to_numpy(),
            "churn_probability": probabilities,
            "risk_band": bands,
        }
    )
    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def score_customer_cohort(
    pipeline: Any,
    customer_data: pd.DataFrame,
    high_risk_threshold: float = 0.65,
) -> pd.DataFrame:
    """Score a cohort and attach useful context fields to the review queue."""
    identifiers = customer_data.copy()
    if "customer_id" not in identifiers:
        identifiers.insert(
            0,
            "customer_id",
            [f"customer-{position:05d}" for position in range(1, len(identifiers) + 1)],
        )
    else:
        ids = identifiers["customer_id"].astype("string").str.strip()
        used_ids = set(ids[ids.notna() & ids.ne("")].tolist())
        normalized_ids = []
        for position, customer_id in enumerate(ids.tolist(), start=1):
            if pd.isna(customer_id) or not customer_id:
                candidate = f"customer-{position:05d}"
                suffix = 1
                while candidate in used_ids:
                    candidate = f"customer-{position:05d}-{suffix}"
                    suffix += 1
                used_ids.add(candidate)
                normalized_ids.append(candidate)
            else:
                normalized_ids.append(str(customer_id))
        identifiers["customer_id"] = normalized_ids

    scored = predict_customers(pipeline, identifiers, high_risk_threshold)
    lookup = identifiers.set_index("customer_id")
    for column in ["contract", "tenure_months", "monthly_charges", "actual_churn"]:
        if column in lookup.columns:
            scored[column] = scored["customer_id"].map(lookup[column])
    return scored


def score_holdout_customers(
    pipeline: Any,
    holdout_data: pd.DataFrame,
    high_risk_threshold: float = 0.65,
) -> pd.DataFrame:
    """Compatibility wrapper for the default holdout review queue."""
    return score_customer_cohort(pipeline, holdout_data, high_risk_threshold)


def profile_signals(customer: pd.Series) -> list[str]:
    """Describe observable profile attributes without implying causal explanations."""
    signals = []
    if customer["contract"] == "Month-to-month":
        signals.append("Month-to-month contract")
    if customer["tenure_months"] < 12:
        signals.append("Less than 12 months of tenure")
    if customer["support_tickets_90d"] >= 3:
        signals.append("Three or more recent support tickets")
    if not bool(customer["autopay"]):
        signals.append("Autopay is not enabled")
    return signals or ["No predefined profile flags"]