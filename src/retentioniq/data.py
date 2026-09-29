"""Demo-data generation and input validation for RetentionIQ."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "contract",
    "internet_service",
    "support_tickets_90d",
    "payment_method",
    "paperless_billing",
    "autopay",
    "senior_citizen",
]
TARGET_COLUMN = "churn"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def generate_demo_data(n_samples: int = 2400, random_state: int = 42) -> pd.DataFrame:
    """Return deterministic synthetic records for an offline product demo."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 to create a useful demo dataset.")

    rng = np.random.default_rng(random_state)
    tenure = rng.integers(0, 73, size=n_samples)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.56, 0.24, 0.20],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.48, 0.18])
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_samples,
        p=[0.36, 0.25, 0.22, 0.17],
    )
    tickets = rng.poisson(1.0, size=n_samples).clip(0, 8)
    senior = rng.binomial(1, 0.17, size=n_samples)
    paperless = rng.binomial(1, 0.60, size=n_samples)
    autopay = rng.binomial(1, 0.48, size=n_samples)
    charges = np.clip(rng.normal(68, 22, size=n_samples), 18, 125).round(2)

    log_odds = (
        -2.15
        + 1.10 * (contract == "Month-to-month")
        + 0.48 * (contract == "One year")
        + 0.42 * (internet == "Fiber optic")
        + 0.21 * tickets
        + 0.012 * (charges - 65)
        - 0.035 * tenure
        - 0.62 * autopay
        + 0.30 * senior
        + 0.18 * (payment == "Electronic check")
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))

    return pd.DataFrame(
        {
            "customer_id": [f"C{index:06d}" for index in range(1, n_samples + 1)],
            "tenure_months": tenure,
            "monthly_charges": charges,
            "contract": contract,
            "internet_service": internet,
            "support_tickets_90d": tickets,
            "payment_method": payment,
            "paperless_billing": paperless,
            "autopay": autopay,
            "senior_citizen": senior,
            "churn": rng.binomial(1, churn_probability).astype(bool),
        }
    )


def validate_dataset(data: pd.DataFrame, minimum_rows: int = 100) -> pd.DataFrame:
    """Normalize supported target values and reject unusable training tables."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if len(data) < minimum_rows:
        raise ValueError(f"Dataset must contain at least {minimum_rows} rows.")

    columns = (["customer_id"] if "customer_id" in data.columns else []) + REQUIRED_COLUMNS
    clean = data[columns].copy()
    target = clean[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError("The churn target cannot contain missing values.")

    if target.dtype == object or pd.api.types.is_string_dtype(target):
        normalized = target.astype(str).str.strip().str.lower()
        target_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        if not normalized.isin(target_map).all():
            raise ValueError("Churn values must be yes/no, true/false, or 1/0.")
        clean[TARGET_COLUMN] = normalized.map(target_map).astype(int)
    else:
        values = set(target.unique())
        if not values.issubset({0, 1, False, True}):
            raise ValueError("Churn values must be binary (0/1 or true/false).")
        clean[TARGET_COLUMN] = target.astype(int)

    if clean[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The churn target must contain both classes.")
    if clean[TARGET_COLUMN].value_counts().min() < 4:
        raise ValueError("The churn target must contain at least four rows in each class for a holdout split.")

    for column in FEATURE_COLUMNS:
        if column in {"contract", "internet_service", "payment_method"}:
            clean[column] = clean[column].where(clean[column].isna(), clean[column].astype(str))
        else:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean