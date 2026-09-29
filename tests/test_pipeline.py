import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retentioniq.data import generate_demo_data, validate_dataset
from retentioniq.model import train_model


def test_demo_data_is_reproducible_and_binary():
    first = generate_demo_data(200, random_state=7)
    second = generate_demo_data(200, random_state=7)
    assert first.equals(second)
    assert set(first["churn"].unique()) == {False, True}


def test_validation_accepts_common_yes_no_target():
    data = generate_demo_data(200)
    data["churn"] = data["churn"].map({True: "Yes", False: "No"})
    clean = validate_dataset(data)
    assert set(clean["churn"].unique()) == {0, 1}


def test_validation_rejects_missing_features():
    data = generate_demo_data(200).drop(columns=["autopay"])
    with pytest.raises(ValueError, match="autopay"):
        validate_dataset(data)


def test_validation_rejects_too_few_minority_class_rows():
    data = generate_demo_data(200)
    data["churn"] = False
    data.loc[:2, "churn"] = True
    with pytest.raises(ValueError, match="four rows in each class"):
        validate_dataset(data)


def test_training_returns_probabilistic_scores_and_metrics():
    result = train_model(generate_demo_data(400))
    sample = generate_demo_data(200)
    scores = result["pipeline"].predict_proba(sample[result["pipeline"].feature_names_in_].head(12))[:, 1]
    assert result["model_name"] in {"Logistic Regression", "Random Forest"}
    assert 0 <= result["metrics"]["roc_auc"] <= 1
    assert len(result["comparison"]) == 2
    assert ((scores >= 0) & (scores <= 1)).all()