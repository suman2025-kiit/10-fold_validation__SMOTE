# QuanFraud Validation & SMOTE Pipelines

Two upload-ready scripts for your GitHub repo:

- `quanfraud_nested_cv.py` — leakage-safe **nested 10-fold** validation (outer=10, inner=5) with optional PCA and score calibration.
- `smote_nested_cv.py` — **SMOTE + nested validation** (SMOTE applied *inside* training folds via `imblearn`).

Both export out-of-fold (OOF) predictions (`*.csv`) and a concise metrics summary (`*.json`):
- Balanced Accuracy (BA)
- ROC-AUC
- PR-AUC
- Recall
- F1
- Chosen thresholds / hyperparameters

## Quick start

```bash
# (Optional) create env
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install -U scikit-learn imbalanced-learn pandas numpy
```

### 1) Nested 10-fold validation (QSVC stand-in via SVC-RBF)
```bash
python quanfraud_nested_cv.py   --csv sample.csv   --target fraud_label   --pca_components 16   --calibration platt   --model svc_rbf   --output_dir outputs_nested_cv
```

### 2) SMOTE + nested validation
```bash
python smote_nested_cv.py   --csv sample.csv   --target fraud_label   --pca_components 16   --clf gb   --k_neighbors 5   --output_dir outputs_smote_cv
```

> **Leakage safety:** All preprocessing and SMOTE occur **inside** training folds only. Thresholds are tuned on inner folds, then applied to the held-out fold.

## Input CSV format

Provide numeric features + a binary target (0=legit, 1=fraud). See `sample.csv` for a template.

**Required:**
- Column: `fraud_label` (0/1)
- Any number of numeric feature columns, e.g., `amount`, `txn_time`, `pca_0..pca_k`.

## Outputs

- `outputs_*/oof_predictions*.csv`: per-row `fold, y_true, y_score, y_pred`
- `outputs_*/metrics_summary*.json`: dataset-level metrics and chosen thresholds/HPs

## Notes

- `svc_rbf` approximates QSVC behavior on a classical kernel; swap with a quantum kernel later if desired.
- For severe imbalance, consider `--clf logreg` with `class_weight="balanced"` or tune SMOTE `--k_neighbors`.
- Reproducibility: random seeds fixed to 42.
