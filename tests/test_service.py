import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retentioniq.data import generate_demo_data
from retentioniq.model import train_model
from retentioniq.service import (
    load_model_artifact,
    predict_customers,
    score_holdout_customers,
    train_and_save,
)


def test_prediction_returns_sorted_bands_and_probability_range():
    data = generate_demo_data(400)
    result = train_model(data)
    scored = predict_customers(result["pipeline"], data.drop(columns="churn"), 0.60)
    assert scored["churn_probability"].is_monotonic_decreasing
    assert set(scored["risk_band"]).issubset({"Low", "Watch", "High"})
    assert scored["churn_probability"].between(0, 1).all()


def test_prediction_rejects_incomplete_customer_schema():
    result = train_model(generate_demo_data(400))
    with pytest.raises(ValueError, match="autopay"):
        predict_customers(result["pipeline"], pd.DataFrame({"tenure_months": [5]}))


def test_prediction_generates_identifier_when_missing():
    data = generate_demo_data(400)
    result = train_model(data)
    data.loc[data.index[0], "customer_id"] = None
    scored = predict_customers(result["pipeline"], data.drop(columns="churn"))
    assert "customer-00001" in set(scored["customer_id"])


def test_prediction_rejects_duplicate_customer_ids():
    data = generate_demo_data(400)
    result = train_model(data)
    scoring_data = data.drop(columns="churn").head(2).copy()
    scoring_data.loc[scoring_data.index[1], "customer_id"] = scoring_data.iloc[0]["customer_id"]
    with pytest.raises(ValueError, match="must be unique"):
        predict_customers(result["pipeline"], scoring_data)


def test_holdout_review_queue_supports_training_data_without_customer_id():
    training_data = generate_demo_data(400).drop(columns="customer_id")
    result = train_model(training_data)
    scored = score_holdout_customers(result["pipeline"], result["holdout_data"])
    assert scored["customer_id"].is_unique
    assert scored["actual_churn"].notna().all()
    assert scored["contract"].notna().all()


def test_model_artifact_round_trip(tmp_path):
    destination = tmp_path / "nested" / "model.joblib"
    saved = train_and_save(generate_demo_data(400), destination)
    loaded = load_model_artifact(destination)
    assert loaded["model_name"] == saved["model_name"]
    assert loaded["schema_version"] == 1
    assert loaded["pipeline"].predict_proba(
        generate_demo_data(200).drop(columns="churn").head(2)[loaded["feature_columns"]]
    ).shape == (2, 2)