from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retentioniq.data import FEATURE_COLUMNS, generate_demo_data, validate_dataset
from retentioniq.model import train_model
from retentioniq.service import predict_customers, profile_signals, score_customer_cohort

st.set_page_config(
    page_title="RetentionIQ | Churn intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)


@st.cache_resource(max_entries=4)
def train_cached(source_bytes: bytes | None) -> dict:
    training_data = (
        generate_demo_data()
        if source_bytes is None
        else pd.read_csv(BytesIO(source_bytes))
    )
    return train_model(training_data)


def format_percent(value: float) -> str:
    return f"{value:.1%}"


st.title("RetentionIQ", icon=":material/query_stats:")
st.caption("Customer churn risk and retention analytics")

with st.sidebar:
    st.subheader("Workspace")
    source_mode = st.segmented_control(
        "Training dataset",
        ["Synthetic demo", "Upload CSV"],
        default="Synthetic demo",
        key="source_mode",
    )
    training_upload = None
    if source_mode == "Upload CSV":
        training_upload = st.file_uploader("Labeled training CSV", type=["csv"], key="training_csv")
    scoring_upload = st.file_uploader("New customers to score (optional)", type=["csv"], key="scoring_csv")
    high_risk_threshold = st.slider(
        "High-risk review threshold",
        min_value=0.25,
        max_value=0.75,
        value=0.35,
        step=0.05,
        help="Customers at or above this predicted probability enter the high-risk band.",
    )
    sample_data = generate_demo_data(300)
    st.download_button(
        "Download sample CSV",
        data=sample_data.to_csv(index=False).encode("utf-8"),
        file_name="retentioniq_synthetic_demo.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption("Educational prototype · No customer decisions")

if source_mode == "Upload CSV" and training_upload is None:
    st.info("Upload a labeled CSV with the documented training columns to begin.")
    st.stop()

source_bytes = training_upload.getvalue() if training_upload is not None else None
try:
    training_data = (
        generate_demo_data()
        if source_bytes is None
        else validate_dataset(pd.read_csv(BytesIO(source_bytes)))
    )
    with st.spinner("Preparing holdout evaluation and comparing baseline models..."):
        result = train_cached(source_bytes)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not prepare this dataset: {exc}")
    st.stop()

observed_churn_rate = float(training_data["churn"].astype(int).mean())
metric_row = st.container(horizontal=True)
with metric_row:
    st.metric("Customers analyzed", f"{len(training_data):,}", border=True)
    st.metric("Observed churn", format_percent(observed_churn_rate), border=True)
    st.metric("Holdout ROC-AUC", f"{result['metrics']['roc_auc']:.3f}", border=True)
    st.metric("Selected model", result["model_name"], border=True)

if source_bytes is None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Synthetic demo data", icon=":material/science:", color="orange")
        st.caption("Metrics are illustrative only and do not represent real-world performance.")
else:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Uploaded training data", icon=":material/upload_file:", color="blue")
        st.caption(
            f"Stratified 25% holdout · {result['test_size']:,} records · "
            "Not production validation."
        )

portfolio_tab, scoring_tab, evaluation_tab = st.tabs(
    ["Portfolio view", "Score customers", "Model evaluation"]
)

with portfolio_tab:
    st.header("Retention portfolio")
    contract_summary = (
        training_data.assign(churn_value=training_data["churn"].astype(int))
        .groupby("contract", as_index=False)
        .agg(customers=("churn_value", "size"), churn_rate=("churn_value", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    tenure_summary = training_data.assign(
        tenure_band=pd.cut(
            training_data["tenure_months"],
            bins=[-1, 6, 12, 24, 48, 120],
            labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
        ),
        churn_value=training_data["churn"].astype(int),
    ).groupby("tenure_band", observed=False, as_index=False).agg(
        customers=("churn_value", "size"), churn_rate=("churn_value", "mean")
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Churn rate by contract")
        st.bar_chart(contract_summary, x="contract", y="churn_rate", y_label="Observed churn rate")
    with chart_right:
        st.subheader("Churn rate by tenure")
        st.bar_chart(tenure_summary, x="tenure_band", y="churn_rate", y_label="Observed churn rate")

    if scoring_upload is not None:
        try:
            score_data = pd.read_csv(BytesIO(scoring_upload.getvalue()))
            scored_customers = score_customer_cohort(result["pipeline"], score_data, high_risk_threshold)
            score_source_label = f"Uploaded scoring file: {scoring_upload.name}"
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not score the uploaded customer file: {exc}")
            scored_customers = pd.DataFrame()
            score_source_label = "Scoring file unavailable"
    else:
        scored_customers = score_customer_cohort(
            result["pipeline"], result["holdout_data"], high_risk_threshold
        )
        score_source_label = "Stratified holdout records"

    st.subheader("Retention review queue")
    st.caption(f"{score_source_label} · sorted by predicted churn probability")
    if not scored_customers.empty:
        high_risk_count = int((scored_customers["risk_band"] == "High").sum())
        high_risk_value, risk_weighted_value = st.columns(2)
        high_risk_value.metric("High-risk review", f"{high_risk_count:,}")
        if "monthly_charges" in scored_customers:
            risk_weighted_charges = (
                scored_customers["monthly_charges"] * scored_customers["churn_probability"]
            ).sum()
            risk_weighted_value.metric("Risk-weighted monthly charges", f"${risk_weighted_charges:,.0f}")
        st.caption("Risk-weighted charges are an exploratory prioritization proxy, not a revenue forecast.")
        st.dataframe(
            scored_customers.head(20),
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.0f%%"
                ),
                "actual_churn": st.column_config.CheckboxColumn("Observed churn"),
                "monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            },
        )
        st.download_button(
            "Export scored cohort",
            data=scored_customers.to_csv(index=False).encode("utf-8"),
            file_name="retentioniq_scored_customers.csv",
            mime="text/csv",
            icon=":material/download:",
        )

with scoring_tab:
    st.header("Single-customer risk assessment")
    st.caption("The profile flags below describe input values; they are not causal explanations.")
    with st.form("customer_profile"):
        input_left, input_middle, input_right = st.columns(3)
        with input_left:
            tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=120, value=8)
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=10000.0, value=79.5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with input_middle:
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            support_tickets_90d = st.number_input("Support tickets in last 90 days", min_value=0, max_value=100, value=3)
            payment_method = st.selectbox(
                "Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
            )
        with input_right:
            paperless_billing = st.checkbox("Paperless billing", value=True)
            autopay = st.checkbox("Autopay enabled", value=False)
            senior_citizen = st.checkbox("Senior citizen", value=False)
        submitted = st.form_submit_button("Score customer", type="primary", icon=":material/query_stats:")

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "customer_id": "what-if-customer",
                    "tenure_months": tenure_months,
                    "monthly_charges": monthly_charges,
                    "contract": contract,
                    "internet_service": internet_service,
                    "support_tickets_90d": support_tickets_90d,
                    "payment_method": payment_method,
                    "paperless_billing": paperless_billing,
                    "autopay": autopay,
                    "senior_citizen": int(senior_citizen),
                }
            ]
        )
        prediction = predict_customers(result["pipeline"], customer, high_risk_threshold).iloc[0]
        probability_column, risk_column = st.columns(2)
        probability_column.metric("Estimated churn probability", format_percent(prediction["churn_probability"]))
        risk_column.metric("Review band", prediction["risk_band"])
        st.write("Profile indicators to review")
        st.write(" · ".join(profile_signals(customer.iloc[0])))
        st.caption("This score is an estimate, not a certainty or a recommendation to deny service.")

with evaluation_tab:
    st.header("Model evaluation")
    comparison = pd.DataFrame(result["comparison"]).drop(columns="confusion_matrix")
    st.subheader("Baseline comparison")
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column.replace("_", " ").title(), format="%.3f")
            for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        },
    )

    confusion_left, importance_right = st.columns(2)
    with confusion_left:
        st.subheader("Selected model · holdout confusion matrix")
        matrix = pd.DataFrame(
            result["metrics"]["confusion_matrix"],
            index=["Actual stay", "Actual churn"],
            columns=["Predicted stay", "Predicted churn"],
        )
        st.dataframe(matrix, hide_index=False)
        st.caption("Default classification threshold: 0.50. Operational thresholds should be set using business costs.")
    with importance_right:
        st.subheader("Permutation importance")
        importance = pd.DataFrame(result["feature_importance"])
        st.bar_chart(importance, x="feature", y="importance", y_label="Change in holdout ROC-AUC")
        st.caption("Global feature importance is not a causal explanation for an individual customer.")

    st.download_button(
        "Download evaluation summary",
        data=pd.DataFrame(result["comparison"]).to_json(orient="records", indent=2),
        file_name="retentioniq_evaluation.json",
        mime="application/json",
        icon=":material/download:",
    )

st.caption("RetentionIQ v1.0 · Educational prototype · Review data governance and fairness before any real deployment")