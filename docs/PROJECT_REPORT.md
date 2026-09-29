# RetentionIQ: Customer Churn Risk and Retention Analytics

## Abstract

Customer churn can reduce recurring revenue and make retention planning harder. RetentionIQ is an educational data-science prototype that validates customer tables, compares baseline classification models, evaluates predictions on a holdout set, summarizes churn patterns, and demonstrates batch and single-customer risk scoring. The prototype combines a scikit-learn pipeline, Streamlit dashboard, and FastAPI inference service.

The bundled data generator creates synthetic records for an offline demonstration. It does not represent a real company or customer population. Real-data performance and business impact must be measured independently before any such claim is made.

## 1. Problem statement

Retention teams need a way to prioritize limited outreach capacity. This project asks whether selected customer attributes can support a ranking of customers by observed churn risk, and how the ranking changes across common model baselines. The intended output is a review aid, not an automatic customer decision.

## 2. Objectives

1. Build a repeatable data validation and preprocessing pipeline.
2. Compare Logistic Regression and Random Forest classifiers.
3. Evaluate both models on the same stratified holdout partition.
4. Report threshold and ranking metrics, plus a confusion matrix.
5. Demonstrate portfolio summaries, risk scoring, and an inference API.
6. Document data limitations, privacy considerations, and a path to real evaluation.

## 3. Tools and architecture

Python 3.13, pandas, NumPy, scikit-learn, Streamlit, FastAPI, Pydantic, pytest, and Docker are used. The application separates data validation (`src/retentioniq/data.py`), training (`model.py`), artifact/scoring services (`service.py`), the dashboard (`app.py`), and API routes (`api.py`).

## 4. Data

The demo generator simulates tenure, charges, contract type, service, support-ticket count, payment method, billing preferences, and a binary churn label. Its label is sampled from an illustrative rule with noise. It is not sourced from Apex Planet or a real organization.

For a real experiment, document:

| Item | Record before submission |
| --- | --- |
| Dataset title and source | `[fill in]` |
| License / terms of use | `[fill in]` |
| Collection period and unit of analysis | `[fill in]` |
| Number of rows and churn prevalence | `[fill in]` |
| Target definition and prediction horizon | `[fill in]` |
| Missingness, duplicates, and exclusions | `[fill in]` |
| Privacy / consent / anonymization steps | `[fill in]` |

The expected input columns and accepted labels are specified in [DATA_SCHEMA.md](DATA_SCHEMA.md).

## 5. Methodology

The target is binary churn. Customer identifiers are excluded from model features. Numeric columns are median-imputed and standardized; categorical columns are most-frequent-imputed and one-hot encoded. The preprocessing steps are fitted only on training rows within each scikit-learn pipeline.

The default experiment uses a reproducible, stratified 75/25 train/holdout split with random state 42. Logistic Regression provides a linear baseline; Random Forest provides a nonlinear tree-ensemble comparison. The candidate with higher holdout ROC-AUC is selected for the demo risk queue. Permutation importance is computed using holdout rows.

Metrics reported are accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix at a 0.50 classification threshold. These metrics answer different questions: ROC-AUC/PR-AUC assess ranking, while precision/recall/F1 and the confusion matrix depend on a classification threshold. The dashboard's operational review threshold is adjustable and does not alter the reported 0.50 confusion matrix.

## 6. Results

Do not paste synthetic demo scores here as evidence about real customers. Run the project on a properly licensed dataset, save the dataset version and split configuration, and fill this table from `retentioniq_evaluation.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `[measure]` | `[measure]` | `[measure]` | `[measure]` | `[measure]` | `[measure]` |
| Random Forest | `[measure]` | `[measure]` | `[measure]` | `[measure]` | `[measure]` | `[measure]` |

### Interpretation prompts

- Which model ranked churn cases better, and was the difference meaningful across repeated splits?
- How did precision and recall change at the chosen intervention threshold?
- What did the confusion matrix imply for outreach capacity and missed cases?
- Were performance and calibration consistent across time periods and relevant customer groups?
- Did high permutation importance align with plausible business context, without being treated as causal evidence?

## 7. Validation plan for a stronger study

1. Define the prediction date and churn window before feature extraction.
2. Split by time, and group repeated observations by customer where appropriate.
3. Compare against a majority-class and simple business-rule baseline.
4. Tune thresholds using documented outreach capacity and error costs, not holdout labels.
5. Report confidence intervals or repeated-fold distributions, calibration, and subgroup metrics.
6. Keep a final untouched test period for one-time evaluation.
7. Record dataset version, code revision, dependency versions, random seed, and model artifact hash.

## 8. Limitations and ethics

The synthetic labels are generated from illustrative assumptions; models can learn those assumptions rather than general customer behavior. Random splitting may overstate future performance when behavior changes over time. Probability values are not calibrated guarantees. Permutation importance is global and associational, while profile flags are descriptive. Age-related or other demographic variables and their proxies require explicit necessity, fairness, and legal review. A human should review any outreach action and the customer should not be denied service on the basis of this score.

## 9. Conclusion

RetentionIQ demonstrates an end-to-end, testable Python workflow for churn analysis and inference. It provides a foundation for an academic experiment, not a production claim. The principal next step is a governed, time-aware evaluation on a real dataset with documented data rights, privacy review, calibration, subgroup analysis, and business-cost validation.

## 10. Contribution statement

Replace this section with each team member's actual work. Do not claim work that was not performed.

- Data preparation and validation: `[name / work completed]`
- Modeling and evaluation: `[name / work completed]`
- Dashboard and API: `[name / work completed]`
- Report and presentation: `[name / work completed]`