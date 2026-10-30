# Churn model — threshold decision

## Business objective
The stated operating preference is to prioritize churner detection over minimizing outreach volume. Missing a real churner (false negative) is treated as more costly than contacting a customer who would have stayed (false positive). The operational recall service level is **80%**.

## How the threshold was selected
- Candidate model family: **scale_pos_weight**, chosen because it had the stronger cross-validated average precision during tuning (CV PR-AUC: **0.6601**).
- Threshold selection data: stratified out-of-fold predicted probabilities on the training partition only. The final hold-out labels were not used to choose the threshold.
- Rule: among OOF thresholds achieving at least 80% recall, choose the one with the highest F1 (and then precision as tie-breaker).
- Selected threshold: **0.4547**, versus the conventional default **0.5000**.
- At that OOF operating point: precision **0.508**, recall **0.814**.
- Rationale: Selected the threshold with the highest F1 among OOF thresholds achieving recall >= 80%.

## Untouched hold-out check
At threshold 0.4547, hold-out precision was **0.501**, recall **0.813**, F1 **0.620**, ROC-AUC **0.843**, and PR-AUC **0.655**. These are the final evaluation estimates, not inputs to threshold selection.

## Deployment implications
A lower threshold generally catches more churners but increases false positives and retention-contact costs. Before deployment, estimate the cost of an intervention, the value of a saved customer, and contact capacity; then choose a threshold against expected business value. Monitor precision, recall, calibration, and drift after launch. Revisit the threshold when campaign capacity or economics change.

See `precision_recall_curve.png` for the training OOF precision–recall curve and `model_comparison.csv` for the two imbalance strategies at the default threshold and the selected model at the chosen threshold.
