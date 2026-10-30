# Churn model — structured error analysis

## Evaluation setup
- Deduplicated rows: 7,043 (removed 0 duplicate customer rows from 7,043).
- Leakage columns removed: none present.
- `customerID` excluded from predictors; `TotalCharges` coerced to numeric.
- Untouched stratified hold-out: 1,409 customers; selected strategy: **scale_pos_weight**.
- Operating threshold: **0.4547** (selected using training-only out-of-fold probabilities).

## Hold-out performance at selected threshold
- Precision: 0.501
- Recall: 0.813
- F1: 0.620
- ROC-AUC: 0.843
- PR-AUC: 0.655
- Confusion matrix (rows actual [No, Yes], columns predicted [No, Yes]): `[[732, 303], [70, 304]]`

## Ten hold-out mistakes
The table is anonymized: customer IDs are intentionally omitted. It includes up to five false negatives and five false positives (or all available errors if fewer).

|   holdout_row | Contract       |   tenure |   MonthlyCharges | InternetService   | PaymentMethod             | TechSupport   | OnlineSecurity   | PaperlessBilling   |   SeniorCitizen |   actual |   predicted | error_type     |   probability_churn |
|--------------:|:---------------|---------:|-----------------:|:------------------|:--------------------------|:--------------|:-----------------|:-------------------|----------------:|---------:|------------:|:---------------|--------------------:|
|           607 | Month-to-month |       35 |           100.5  | Fiber optic       | Bank transfer (automatic) | Yes           | Yes              | Yes                |               1 |        1 |           0 | False negative |            0.445703 |
|           103 | Month-to-month |       34 |           109.8  | Fiber optic       | Mailed check              | No            | Yes              | Yes                |               0 |        1 |           0 | False negative |            0.444273 |
|           744 | Month-to-month |        4 |            61.45 | DSL               | Credit card (automatic)   | Yes           | No               | No                 |               0 |        1 |           0 | False negative |            0.442189 |
|           717 | One year       |       53 |           106.95 | Fiber optic       | Credit card (automatic)   | No            | Yes              | Yes                |               0 |        1 |           0 | False negative |            0.430345 |
|           874 | Month-to-month |        5 |            53.85 | DSL               | Credit card (automatic)   | Yes           | Yes              | No                 |               0 |        1 |           0 | False negative |            0.427881 |
|           733 | Month-to-month |        1 |            79.15 | Fiber optic       | Mailed check              | No            | No               | Yes                |               0 |        0 |           1 | False positive |            0.951895 |
|            41 | Month-to-month |        2 |            91.45 | Fiber optic       | Electronic check          | No            | No               | Yes                |               0 |        0 |           1 | False positive |            0.938688 |
|           889 | Month-to-month |        2 |            84.05 | Fiber optic       | Electronic check          | No            | No               | Yes                |               1 |        0 |           1 | False positive |            0.937451 |
|           995 | Month-to-month |        7 |            94.1  | Fiber optic       | Electronic check          | No            | No               | Yes                |               0 |        0 |           1 | False positive |            0.936423 |
|          1222 | Month-to-month |        1 |            74.3  | Fiber optic       | Electronic check          | No            | No               | No                 |               1 |        0 |           1 | False positive |            0.931971 |

## Shared-characteristic scan
- **Contract:** False positive / Month-to-month: 5; False negative / Month-to-month: 4; False negative / One year: 1
- **tenure_band:** False positive / 0–12: 5; False negative / 0–12: 2; False negative / 25–48: 2; False negative / 49–72: 1
- **InternetService:** False positive / Fiber optic: 5; False negative / Fiber optic: 3; False negative / DSL: 2
- **PaymentMethod:** False positive / Electronic check: 4; False negative / Credit card (automatic): 3; False negative / Bank transfer (automatic): 1; False negative / Mailed check: 1; False positive / Mailed check: 1
- **TechSupport:** False positive / No: 5; False negative / Yes: 3; False negative / No: 2
- **OnlineSecurity:** False positive / No: 5; False negative / Yes: 4; False negative / No: 1

## Interpretation and limitations
- False negatives are churners missed by the model; these are costly when retention outreach could have prevented a cancellation.
- False positives are customers flagged as churn risks who did not churn; these can waste retention incentives.
- Review the category counts alongside the ten-row table. A pattern in ten errors is exploratory, not proof of causation; validate it on a larger future cohort before changing policy.
- `misclassified_customers.csv` contains the same anonymized cases in machine-readable form.