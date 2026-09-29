# Data schema

The dashboard accepts CSV files. Training data requires at least 100 rows, all feature columns below, a binary `churn` label, and both target classes. Target labels may be `yes`/`no`, `true`/`false`, or `1`/`0` (case-insensitive for text labels). A `customer_id` column is optional and is excluded from model features.

| Column | Type | Meaning |
| --- | --- | --- |
| `customer_id` | string, optional | Pseudonymous row identifier; never used as a model feature |
| `tenure_months` | number | Months since customer activation |
| `monthly_charges` | number | Current recurring monthly charge |
| `contract` | category | `Month-to-month`, `One year`, or `Two year` |
| `internet_service` | category | `DSL`, `Fiber optic`, or `No` |
| `support_tickets_90d` | number | Support tickets in the previous 90 days |
| `payment_method` | category | `Electronic check`, `Credit card`, `Bank transfer`, or `Mailed check` |
| `paperless_billing` | boolean / 0 or 1 | Paperless billing enrollment |
| `autopay` | boolean / 0 or 1 | Automatic payment enrollment |
| `senior_citizen` | 0 or 1 | Broad demographic indicator; review necessity and fairness before real use |
| `churn` | binary target | Whether the customer left during the observation window; required only for training/evaluation |

For the **new customers to score** upload, provide the nine feature columns and optionally `customer_id`. The `churn` target is not required. Unknown extra columns are ignored by model prediction. Numeric missing values are median-imputed and categorical missing values are most-frequent-imputed within the training pipeline; malformed numeric values are treated as missing.

The bundled generator produces simulated values and labels. It is for interface and code-path demonstrations only, not evidence about a real business or population.