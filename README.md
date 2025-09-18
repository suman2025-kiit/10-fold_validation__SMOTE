# 10-fold_validation__SMOTE
Detailed step for our "QuanFraud" scheme specially for 10-fold_validation approach and SMOTE (Synthetic Minority Oversampling Technique 'SMOTE' is a statistical technique for increasing the number of cases in your dataset in a balanced way.)

quantum-blockchain-/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sample.csv
├─ scripts/
│  ├─ quanfraud_nested_cv.py
│  └─ smote_nested_cv.py
└─ outputs/               

# will be generated automatically by scripts as mentioned below for validation for "QuanFraud" scheme 

## Requirements

Create requirements.txt:
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.12

## Create and activate a virtual environment:

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

## Input data format

A CSV (attached only for sample) with numeric feature columns and a binary target (0 = legit, 1 = fraud). Example (data/sample.csv)

## Script 1 — Nested 10-Fold Validation
File: scripts/quanfraud_nested_cv.py (attached herewith)

What it does:

Outer 10 folds for out-of-fold (OOF) evaluation

Inner 5 folds for hyperparameter & threshold tuning

Preprocessing inside CV folds: impute → scale → optional PCA

(Optional) probability calibration (Platt/Isotonic)

Metrics: Balanced Accuracy, ROC-AUC, PR-AUC, Recall, F1

Saves: oof_predictions.csv and metrics_summary.json

Run (example):

python scripts/quanfraud_nested_cv.py \
  --csv data/sample.csv \
  --target fraud_label \
  --pca_components 16 \
  --calibration platt \
  --model svc_rbf \
  --output_dir outputs/nested_cv


Key flags:

--pca_components 16 → set PCA dimension (omit to disable PCA)

--calibration {none,platt,isotonic} → probability calibration

--model {svc_rbf,logreg} → SVC (RBF) ≈ classical stand-in for QSVC; logistic regression alternative

--outer_folds 10 --inner_folds 5 → fold counts

Outputs:

outputs/nested_cv/oof_predictions.csv → fold, y_true, y_score, y_pred

outputs/nested_cv/metrics_summary.json → metrics + chosen thresholds/HPs

## Script 2 — SMOTE + Nested Validation (leakage-safe)

File: scripts/smote_nested_cv.py
What it does:

Same nested CV design as above

SMOTE is applied inside training folds only (no leakage)

Classifiers: GradientBoosting (default) or LogisticRegression

Saves: oof_predictions_smote.csv, metrics_summary_smote.json

Run (example):

python scripts/smote_nested_cv.py \
  --csv data/sample.csv \
  --target fraud_label \
  --pca_components 16 \
  --clf gb \
  --k_neighbors 5 \
  --output_dir outputs/smote_cv


Key flags:

--clf {gb,logreg} → choose classifier

--k_neighbors 5 → SMOTE neighbor count

--pca_components → same as above

Outputs:

outputs/smote_cv/oof_predictions_smote.csv

outputs/smote_cv/metrics_summary_smote.json

## What are the metrics mean

Balanced Accuracy (BA): robust when classes are imbalanced

ROC-AUC: ranking quality across thresholds

PR-AUC: better visibility under heavy imbalance

Recall: fraud detection sensitivity

F1: balance of precision and recall

Threshold is tuned inside the inner folds to maximize F1 (default), then applied to the outer test fold.

## Quick run
### python scripts/quanfraud_nested_cv.py \
  --csv data/sample.csv --target fraud_label \
  --pca_components 16 --calibration platt \
  --model svc_rbf --output_dir outputs/nested_cv

### python scripts/smote_nested_cv.py \
  --csv data/sample.csv --target fraud_label \
  --pca_components 16 --clf gb \
  --k_neighbors 5 --output_dir outputs/smote_cv

## Outputs

oof_predictions*.csv – out-of-fold scores/preds

metrics_summary*.json – Balanced Accuracy, ROC-AUC, PR-AUC, Recall, F1, chosen thresholds/hyperparams

All preprocessing and SMOTE occur inside the training folds to prevent data leakage.
