# Telco Customer Churn Prediction

An end-to-end machine learning workflow for predicting telecom customer churn using **XGBoost**. The project combines leakage-aware preprocessing, class-imbalance handling, cross-validated hyperparameter tuning, business-driven threshold selection, and structured error analysis.

> **Project focus:** A useful churn model is not just one with a strong score. It should also make sensible trade-offs between missed churners and unnecessary retention outreach—and its mistakes should be inspected.

## Project overview

Customer churn prediction helps a telecom business identify customers who may leave so that retention teams can prioritize timely interventions. Churn is an imbalanced binary classification problem, so this project evaluates more than accuracy and compares two imbalance-handling approaches.

The workflow:

1. Cleans the data and removes duplicate customer records before splitting.
2. Excludes customer identifiers and known leakage columns from model features.
3. Builds preprocessing and classification into one pipeline.
4. Tunes an XGBoost classifier with stratified cross-validation.
5. Compares class weighting (`scale_pos_weight`) with SMOTE.
6. Selects a decision threshold using training-only out-of-fold predictions.
7. Evaluates the selected model once on an untouched stratified hold-out set.
8. Examines false negatives and false positives for recurring patterns.

## Dataset

The script expects a Telco customer churn CSV with these columns:

- `customerID` — customer identifier; excluded from model predictors.
- `Churn` — target label (`Yes` / `No`).
- Customer account, service, billing, and demographic attributes used as candidate predictors.

By default, the expected filename is:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Place the CSV in the project directory, or pass its path as the first command-line argument. The dataset is not bundled in this repository package; obtain it from the source provided for your course/project and follow its usage terms.

## Methodology

### Data quality and leakage prevention

- Duplicate records are removed using `customerID` before the train/test split.
- `customerID` and `Churn` are excluded from the feature matrix.
- Known leakage fields—`churn_risk_score` and `post_outcome_churn_flag`—are removed when present.
- `TotalCharges` is converted to numeric; invalid or blank values are handled by the numeric imputer.
- The data is split into 80% training and 20% hold-out data using stratification and a fixed random seed.

### Unified machine-learning pipeline

The fitted workflow uses an `imblearn` pipeline containing:

- **Numeric preprocessing:** median imputation and standard scaling.
- **Categorical preprocessing:** most-frequent imputation and one-hot encoding.
- **Imbalance stage:** either SMOTE within the training fold or passthrough for class weighting.
- **Classifier:** `XGBClassifier` with tuned hyperparameters.

Keeping preprocessing and sampling inside the pipeline ensures that transformations and SMOTE are fitted within each training fold rather than on validation or hold-out data.

### Hyperparameter tuning

`RandomizedSearchCV` optimizes average precision (area under the precision–recall curve) over seven XGBoost hyperparameters:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`

Two imbalance strategies are compared:

1. **Class weighting:** `scale_pos_weight`, calculated from the training partition.
2. **SMOTE:** synthetic minority oversampling applied within the pipeline.

If the cross-validation average-precision scores differ by less than 0.001, the script treats them as practically tied and prefers `scale_pos_weight` because it avoids synthetic samples and is simpler to monitor.

### Business-driven threshold selection

The operating preference is to prioritize identifying churners. The threshold is selected from stratified out-of-fold probabilities generated on the training partition only:

- Target recall: **at least 80%**
- Selection rule: among thresholds meeting that recall target, choose the one with the highest F1 score (precision is the tie-breaker).
- The hold-out labels are not used to select the threshold.

This makes the threshold a business decision rather than automatically accepting the conventional 0.50 cutoff. A lower threshold can catch more churners, but usually increases false-positive retention contacts.

## Results

The following results are from the included run on **7,043 customer records**, with an untouched **1,409-row hold-out set**. The run used 4 randomized candidates per strategy and 2 cross-validation folds for practical execution.

### Hold-out comparison

| Strategy | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Class weighting (`scale_pos_weight`) | 0.5000 | 0.521 | 0.799 | 0.631 | 0.845 | 0.666 |
| SMOTE | 0.5000 | 0.571 | 0.655 | 0.610 | 0.841 | 0.646 |
| **Selected class-weighted model** | **0.4768** | **0.514** | **0.813** | **0.630** | **0.845** | **0.666** |

The selected threshold achieved **81.3% recall** on the hold-out set, meeting the intended recall-oriented operating preference. The confusion matrix at this threshold was:

```text
                 Predicted
                 No    Yes
Actual No        748   287
Actual Yes        70   304
```

These metrics are evaluation results for this particular split and run; they are not a guarantee of future production performance. The included search settings are intentionally modest. For a more thorough tuning run, increase the number of randomized candidates and CV folds as shown below.

## Error analysis

The project produces a structured review of up to ten hold-out mistakes (up to five false negatives and five false positives). Customer IDs are omitted from the error sample.

In the included sample:

- All ten sampled errors involved month-to-month contracts.
- The sampled false positives were all Fiber optic customers.
- Four of the five sampled false positives used electronic check.
- False negatives included both DSL and Fiber optic customers.

These are exploratory observations from a small error sample, not evidence of causation or proof that the model systematically fails for every customer in those groups. See [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) for the anonymized cases, category counts, and limitations.

## Repository contents

```text
.
├── final_churn_model.py              # End-to-end training, tuning, evaluation script
├── fixed_pipeline.py                 # Earlier corrected pipeline reference
├── ERROR_ANALYSIS.md                 # Hold-out error review and pattern scan
├── THRESHOLD_DECISION.md             # Business objective and threshold rationale
├── model_comparison.csv              # Hold-out metric comparison
├── precision_recall_curve.png        # Training out-of-fold PR curve
├── misclassified_customers.csv       # Anonymized hold-out mistakes
├── best_churn_pipeline.joblib        # Serialized selected pipeline
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Input data (add locally; not included)
```

The output files above are generated by the training script. If you rerun the script, keep the outputs from the same run together because the reports, comparison table, plot, and serialized model are regenerated.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/telco-churn-prediction-ml-pipeline.git
cd telco-churn-prediction-ml-pipeline
```

Replace `<your-username>` with your GitHub username.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib scipy joblib
```

### 4. Add the dataset

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the repository root, or provide the path to your CSV when running the script.

### 5. Run the project

```bash
python final_churn_model.py WA_Fn-UseC_-Telco-Customer-Churn.csv
```

For a more thorough search, use more randomized candidates and cross-validation folds:

```bash
python final_churn_model.py WA_Fn-UseC_-Telco-Customer-Churn.csv --iterations 20 --cv-folds 5
```

The default settings in the script are 4 randomized candidates per strategy and 2 CV folds to keep runtime manageable. Increasing these settings can improve the breadth of the search but will take longer.

## Generated outputs

After a successful run, the script writes:

| Output | Description |
|---|---|
| `best_churn_pipeline.joblib` | Serialized fitted preprocessing + XGBoost pipeline |
| `model_comparison.csv` | Hold-out precision, recall, F1, ROC-AUC, and PR-AUC for both strategies and the selected threshold |
| `precision_recall_curve.png` | Precision–recall curve based on training out-of-fold predictions |
| `ERROR_ANALYSIS.md` | Anonymized hold-out mistakes and exploratory pattern analysis |
| `THRESHOLD_DECISION.md` | Business objective, threshold selection method, and hold-out check |
| `misclassified_customers.csv` | Machine-readable anonymized error examples |

## Reproducibility and responsible use

- A fixed random seed (`42`) is used for the main split and model components.
- Threshold selection uses training-only out-of-fold predictions.
- The hold-out set is kept separate for final evaluation.
- The model should be re-evaluated on newer customer cohorts before operational use.
- Before deploying a threshold, estimate retention-contact costs, customer lifetime value, campaign capacity, and the value of preventing churn.
- Monitor precision, recall, calibration, and data drift after deployment.
- Do not interpret patterns from ten sampled mistakes as causal findings.

## Future improvements

- Expand the randomized hyperparameter search and evaluate stability across repeated splits.
- Add calibration analysis and expected-cost threshold optimization.
- Compare against additional baselines and gradient-boosting implementations.
- Validate error patterns on a larger cohort and monitor subgroup performance.
- Add a reproducible experiment log and model-version tracking.

## License

Add a license file before publishing if you intend others to reuse, modify, or distribute this project. Make sure the dataset's terms also permit your intended use.
