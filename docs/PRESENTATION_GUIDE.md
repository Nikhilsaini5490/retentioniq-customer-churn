# Presentation and viva guide

## Suggested 7-slide flow

1. **Problem and motivation**: retention outreach has limited time and needs a transparent way to prioritize review.
2. **Project objective**: compare baseline classifiers and demonstrate risk summaries; clarify that this is a prototype, not an automatic decision system.
3. **Dataset**: show the source, target definition, prediction window, row count, class balance, and privacy/license notes. If using the built-in dataset, say plainly that it is synthetic.
4. **Method**: explain the train-only preprocessing pipeline, stratified holdout, Logistic Regression baseline, and Random Forest comparison.
5. **Results**: show measured holdout metrics from your documented real-data run. Explain ROC-AUC/PR-AUC separately from thresholded precision and recall.
6. **Product demo**: compare contract/tenure segments, adjust the review threshold, score one sample profile, then show the API docs or batch prediction.
7. **Limitations and next steps**: temporal validation, calibration, subgroup checks, data governance, and human review.

## Three-minute live demo

1. Start `streamlit run app.py` and identify whether the active data source is synthetic or your evaluated real CSV.
2. Point out the data count, observed churn, selected model, and holdout ROC-AUC. State the split and dataset source.
3. Compare the contract and tenure summaries, then explain one model evaluation metric in plain language.
4. Change the high-risk review threshold and explain the capacity/false-positive tradeoff.
5. Score a sample profile; distinguish profile attributes from causal explanations.
6. If time allows, open `/docs` and show the validated single/batch prediction endpoints.

## Likely viva questions

**Why compare these two models?** Logistic Regression is a useful interpretable baseline; Random Forest can represent nonlinear interactions. A more complex model is not automatically better, so both are evaluated on the same holdout.

**Why use ROC-AUC and PR-AUC?** Churn may be a minority class. ROC-AUC measures ranking across thresholds; PR-AUC focuses on precision/recall behavior for the positive class. Neither selects a business threshold by itself.

**What does the score mean?** It is the model-estimated probability under the training distribution. It is not certainty, a causal explanation, or a guaranteed outcome. Calibration must be assessed before interpreting it as a real probability.

**How did you avoid preprocessing leakage?** Imputation, scaling, and one-hot encoding are inside a scikit-learn pipeline fitted on training rows only; the same pipeline transforms holdout rows.

**Why is the supplied data synthetic?** It keeps the demo runnable without redistributing customer data and makes no claim about real behavior. A defensible performance claim requires a licensed real dataset and a documented evaluation.

**What is a false positive here?** A customer predicted to churn who stays. Outreach may then be unnecessary or costly. A false negative is a customer who churns but is not prioritized. Threshold selection should reflect intervention capacity and these costs.

**Can feature importance prove that a feature causes churn?** No. Permutation importance measures how much a model's holdout ranking changes when a feature is shuffled. Correlated features and observational data limit causal interpretation.

**What would you add before production?** Time-based validation, calibration and drift monitoring, subgroup/fairness review, secure authenticated access, audit logs, consent/data-retention controls, model approval, and a human-reviewed intervention process.

## Before presenting

- Run `python -m pytest -q` and keep the passing output for your demo notes.
- Use a real dataset only if its license and academic-use terms allow it.
- Fill the results table in [PROJECT_REPORT.md](PROJECT_REPORT.md) with measured values and the exact split.
- Never use synthetic metrics as claims about Apex Planet, an employer, or real customers.
- Be ready to explain what you personally changed, tested, and learned.