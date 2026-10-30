#!/usr/bin/env python3
"""
Tuned, leakage-aware Telco churn classification.

Usage:
  python final_churn_model.py WA_Fn-UseC_-Telco-Customer-Churn.csv

Outputs:
  ERROR_ANALYSIS.md
  THRESHOLD_DECISION.md
  precision_recall_curve.png
  model_comparison.csv
  best_churn_pipeline.joblib

Requirements:
  pandas, scikit-learn, xgboost, imbalanced-learn, matplotlib, joblib
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TARGET_RECALL = 0.80  # business preference: catch at least 80% of churners


def load_and_clean(path):
    df = pd.read_csv(path)
    required = {"customerID", "Churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    before = len(df)
    df = df.drop_duplicates(subset=["customerID"], keep="first").copy()
    removed_duplicates = before - len(df)
    leakage_cols = ["churn_risk_score", "post_outcome_churn_flag"]
    removed_leakage = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=removed_leakage)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Churn contains values other than Yes/No.")
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    return df, X, y.astype(int), before, removed_duplicates, removed_leakage


def build_pipeline(X, strategy, scale_pos_weight):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")
    sampler = SMOTE(random_state=RANDOM_STATE) if strategy == "smote" else "passthrough"
    clf = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=1, random_state=RANDOM_STATE,
        scale_pos_weight=1.0 if strategy == "smote" else scale_pos_weight,
    )
    return Pipeline([
        ("preprocessor", prep),
        ("sampler", sampler),
        ("classifier", clf),
    ])


def metric_row(name, y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "strategy": name, "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


def choose_threshold(y_true, probabilities, target_recall=TARGET_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # precision/recall have one more element than thresholds.
    candidates = [
        (float(precision[i]), float(recall[i]), float(thresholds[i]))
        for i in range(len(thresholds)) if recall[i] >= target_recall
    ]
    if candidates:
        # Among thresholds meeting the recall service level, maximize F1.
        p, r, t = max(candidates, key=lambda x: (2*x[0]*x[1]/(x[0]+x[1]+1e-12), x[0], x[2]))
        rationale = f"Selected the threshold with the highest F1 among OOF thresholds achieving recall >= {target_recall:.0%}."
    else:
        i = int(np.argmax(recall[:-1]))
        p, r, t = float(precision[i]), float(recall[i]), float(thresholds[i])
        rationale = f"No OOF threshold met recall >= {target_recall:.0%}; selected the threshold with maximum OOF recall."
    return t, p, r, rationale, precision, recall


def describe_patterns(errors, test_df, y_test, pred, probs):
    combined = errors.copy()
    combined["actual"] = np.asarray(y_test)[errors.index]
    combined["predicted"] = np.asarray(pred)[errors.index]
    combined["error_type"] = np.where(combined.actual.eq(1), "False negative", "False positive")
    combined["probability_churn"] = np.asarray(probs)[errors.index]
    combined["tenure_band"] = pd.cut(
        pd.to_numeric(combined.get("tenure"), errors="coerce"),
        bins=[-1, 12, 24, 48, 72], labels=["0–12", "13–24", "25–48", "49–72"]
    )
    rows = []
    for col in ["Contract", "tenure_band", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]:
        if col in combined.columns:
            counts = combined.groupby(["error_type", col], observed=True).size().sort_values(ascending=False)
            top = counts.head(5)
            if len(top):
                rows.append(f"- **{col}:** " + "; ".join(
                    f"{kind} / {value}: {count}" for (kind, value), count in top.items()
                ))
    return combined, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", nargs="?", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--iterations", type=int, default=4,
                        help="RandomizedSearchCV candidates per strategy (default: 8)")
    parser.add_argument("--cv-folds", type=int, default=2)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    df, X, y, before, removed, leakage = load_and_clean(args.input_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    neg, pos = np.bincount(y_train)
    spw = neg / pos
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    distributions = {
        "classifier__n_estimators": randint(80, 221),
        "classifier__max_depth": randint(2, 7),
        "classifier__learning_rate": uniform(0.02, 0.18),
        "classifier__subsample": uniform(0.65, 0.35),
        "classifier__colsample_bytree": uniform(0.65, 0.35),
        "classifier__min_child_weight": randint(1, 8),
        "classifier__gamma": uniform(0.0, 2.0),
    }

    fitted, comparison = {}, []
    for strategy in ("scale_pos_weight", "smote"):
        pipe = build_pipeline(X_train, strategy, spw)
        search = RandomizedSearchCV(
            pipe, distributions, n_iter=args.iterations, scoring="average_precision",
            cv=cv, random_state=RANDOM_STATE, n_jobs=1, refit=True,
            verbose=1, return_train_score=False,
        )
        print(f"\nTuning strategy: {strategy}")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        probs = best.predict_proba(X_test)[:, 1]
        comparison.append(metric_row(strategy, y_test.to_numpy(), probs, 0.5))
        fitted[strategy] = {
            "search": search, "model": best, "test_prob": probs,
            "cv_pr_auc": search.best_score_,
        }
        print("Best CV PR-AUC:", round(search.best_score_, 5))
        print("Best params:", search.best_params_)

    # Select the strategy with best cross-validated PR-AUC, not test-set results.
    # Treat CV PR-AUC differences below 0.001 as practically tied; prefer
    # scale_pos_weight because it avoids synthetic observations and is simpler
    # to monitor operationally. This tie-breaker is fixed before hold-out scoring.
    cv_gap = abs(fitted["scale_pos_weight"]["cv_pr_auc"] - fitted["smote"]["cv_pr_auc"])
    if cv_gap < 0.001:
        selected_strategy = "scale_pos_weight"
    else:
        selected_strategy = max(fitted, key=lambda s: fitted[s]["cv_pr_auc"])
    selected = fitted[selected_strategy]
    model = selected["model"]

    # OOF training probabilities determine threshold; hold-out remains untouched.
    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1)
    oof_prob = cross_val_predict(
        model, X_train, y_train, cv=oof_cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold, oof_p, oof_r, rationale, pr_p, pr_r = choose_threshold(y_train.to_numpy(), oof_prob)

    holdout_prob = selected["test_prob"]
    comparison.append(metric_row(selected_strategy + " (selected threshold)", y_test.to_numpy(),
                                 holdout_prob, threshold))
    pd.DataFrame(comparison).to_csv("model_comparison.csv", index=False)

    # PR curve is generated from out-of-fold training predictions, not test labels.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pr_r, pr_p, label=f"{selected_strategy} OOF PR curve (AP={average_precision_score(y_train, oof_prob):.3f})")
    ax.axvline(oof_r, linestyle="--", alpha=0.65, label=f"Chosen operating point: recall={oof_r:.2f}")
    ax.scatter([oof_r], [oof_p], zorder=3)
    ax.set(xlabel="Recall", ylabel="Precision", title="Telco churn precision–recall curve (training OOF)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig("precision_recall_curve.png", dpi=160); plt.close(fig)

    final_row = metric_row(selected_strategy, y_test.to_numpy(), holdout_prob, threshold)
    pred = (holdout_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    error_mask = pred != y_test.to_numpy()
    test_reset = X_test.reset_index(drop=True)
    error_positions = np.flatnonzero(error_mask)
    # Include up to five FNs and five FPs, then fill remaining slots by confidence.
    fn = [i for i in error_positions if y_test.to_numpy()[i] == 1]
    fp = [i for i in error_positions if y_test.to_numpy()[i] == 0]
    fn = sorted(fn, key=lambda i: holdout_prob[i], reverse=True)[:5]
    fp = sorted(fp, key=lambda i: holdout_prob[i], reverse=True)[:5]
    chosen = fn + fp
    if len(chosen) < min(10, len(error_positions)):
        remaining = [i for i in error_positions if i not in chosen]
        remaining.sort(key=lambda i: abs(holdout_prob[i] - 0.5), reverse=True)
        chosen += remaining[:10-len(chosen)]
    errors = test_reset.iloc[chosen].copy()
    errors.index = chosen
    error_table, patterns = describe_patterns(errors, test_reset, y_test.to_numpy(), pred, holdout_prob)
    safe_cols = [c for c in [
        "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"
    ] if c in error_table.columns]
    display = error_table[safe_cols + ["actual", "predicted", "error_type", "probability_churn"]].copy()
    display.insert(0, "holdout_row", display.index)
    display.to_csv("misclassified_customers.csv", index=False)

    error_md = [
        "# Churn model — structured error analysis", "",
        "## Evaluation setup",
        f"- Deduplicated rows: {len(df):,} (removed {removed:,} duplicate customer rows from {before:,}).",
        f"- Leakage columns removed: {leakage if leakage else 'none present'}.",
        "- `customerID` excluded from predictors; `TotalCharges` coerced to numeric.",
        f"- Untouched stratified hold-out: {len(X_test):,} customers; selected strategy: **{selected_strategy}**.",
        f"- Operating threshold: **{threshold:.4f}** (selected using training-only out-of-fold probabilities).", "",
        "## Hold-out performance at selected threshold",
        f"- Precision: {final_row['precision']:.3f}",
        f"- Recall: {final_row['recall']:.3f}",
        f"- F1: {final_row['f1']:.3f}",
        f"- ROC-AUC: {final_row['roc_auc']:.3f}",
        f"- PR-AUC: {final_row['pr_auc']:.3f}",
        f"- Confusion matrix (rows actual [No, Yes], columns predicted [No, Yes]): `[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]`", "",
        "## Ten hold-out mistakes",
        "The table is anonymized: customer IDs are intentionally omitted. It includes up to five false negatives and five false positives (or all available errors if fewer).",
        "", display.to_markdown(index=False), "",
        "## Shared-characteristic scan",
        *(patterns if patterns else ["- No recurring category pattern could be computed from the selected error rows."]),
        "",
        "## Interpretation and limitations",
        "- False negatives are churners missed by the model; these are costly when retention outreach could have prevented a cancellation.",
        "- False positives are customers flagged as churn risks who did not churn; these can waste retention incentives.",
        "- Review the category counts alongside the ten-row table. A pattern in ten errors is exploratory, not proof of causation; validate it on a larger future cohort before changing policy.",
        "- `misclassified_customers.csv` contains the same anonymized cases in machine-readable form."
    ]
    Path("ERROR_ANALYSIS.md").write_text("\n".join(error_md), encoding="utf-8")

    threshold_md = f"""# Churn model — threshold decision

## Business objective
The stated operating preference is to prioritize churner detection over minimizing outreach volume. Missing a real churner (false negative) is treated as more costly than contacting a customer who would have stayed (false positive). The operational recall service level is **{TARGET_RECALL:.0%}**.

## How the threshold was selected
- Candidate model family: **{selected_strategy}**, chosen because it had the stronger cross-validated average precision during tuning (CV PR-AUC: **{selected['cv_pr_auc']:.4f}**).
- Threshold selection data: stratified out-of-fold predicted probabilities on the training partition only. The final hold-out labels were not used to choose the threshold.
- Rule: among OOF thresholds achieving at least {TARGET_RECALL:.0%} recall, choose the one with the highest F1 (and then precision as tie-breaker).
- Selected threshold: **{threshold:.4f}**, versus the conventional default **0.5000**.
- At that OOF operating point: precision **{oof_p:.3f}**, recall **{oof_r:.3f}**.
- Rationale: {rationale}

## Untouched hold-out check
At threshold {threshold:.4f}, hold-out precision was **{final_row['precision']:.3f}**, recall **{final_row['recall']:.3f}**, F1 **{final_row['f1']:.3f}**, ROC-AUC **{final_row['roc_auc']:.3f}**, and PR-AUC **{final_row['pr_auc']:.3f}**. These are the final evaluation estimates, not inputs to threshold selection.

## Deployment implications
A lower threshold generally catches more churners but increases false positives and retention-contact costs. Before deployment, estimate the cost of an intervention, the value of a saved customer, and contact capacity; then choose a threshold against expected business value. Monitor precision, recall, calibration, and drift after launch. Revisit the threshold when campaign capacity or economics change.

See `precision_recall_curve.png` for the training OOF precision–recall curve and `model_comparison.csv` for the two imbalance strategies at the default threshold and the selected model at the chosen threshold.
"""
    Path("THRESHOLD_DECISION.md").write_text(threshold_md, encoding="utf-8")
    joblib.dump(model, "best_churn_pipeline.joblib")

    print("\nSTRATEGY COMPARISON (hold-out, threshold=0.5 except selected-threshold row)")
    print(pd.DataFrame(comparison).to_string(index=False))
    print("\nSelected strategy:", selected_strategy, "threshold:", round(threshold, 4))
    print("Hold-out selected-threshold metrics:", final_row)
    print("Confusion matrix:\n", cm)
    print("Wrote: ERROR_ANALYSIS.md, THRESHOLD_DECISION.md, precision_recall_curve.png,")
    print("       model_comparison.csv, misclassified_customers.csv, best_churn_pipeline.joblib")


if __name__ == "__main__":
    main()
