"""Reproducible churn-model training and evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retentioniq.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_dataset

CATEGORICAL_COLUMNS = ["contract", "internet_service", "payment_method"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def _make_pipeline(estimator: Any) -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def _metrics(y_true: pd.Series, probabilities: Any) -> dict[str, float | list[list[int]]]:
    predictions = probabilities >= 0.5
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def train_model(data: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    """Compare two baselines and return the best fitted pipeline and holdout results."""
    clean = validate_dataset(data)
    features = clean[FEATURE_COLUMNS]
    target = clean[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    comparison = []
    best_pipeline = None
    best_metrics = None
    best_name = ""

    for name, estimator in candidates.items():
        pipeline = _make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = _metrics(y_test, probabilities)
        comparison.append({"model": name, **metrics})
        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_pipeline = pipeline
            best_metrics = metrics
            best_name = name

    assert best_pipeline is not None and best_metrics is not None
    importance_result = permutation_importance(
        best_pipeline,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=4,
        random_state=random_state,
        n_jobs=-1,
    )
    importances = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(FEATURE_COLUMNS, importance_result.importances_mean, strict=True)
    ]
    importances.sort(key=lambda item: item["importance"], reverse=True)

    holdout_data = x_test.copy()
    if "customer_id" in clean.columns:
        holdout_data.insert(0, "customer_id", clean.loc[x_test.index, "customer_id"])
    holdout_data["actual_churn"] = y_test.astype(bool)

    return {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "metrics": best_metrics,
        "comparison": comparison,
        "feature_importance": importances,
        "test_size": len(y_test),
        "positive_rate": float(y_test.mean()),
        "holdout_data": holdout_data,
    }