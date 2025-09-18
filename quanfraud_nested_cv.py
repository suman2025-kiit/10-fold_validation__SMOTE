#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested 10-fold validation for fraud detection (leakage-safe).
- Outer: 10 stratified folds (out-of-fold evaluation)
- Inner: 5 stratified folds (hyperparameter + threshold selection)
- Preprocessing: impute -> scale -> optional PCA (all fit inside CV)
- Metrics: Balanced Accuracy, ROC-AUC, PR-AUC, Recall, F1
- Optional calibration: Platt (sigmoid) or Isotonic (within training folds only)
- Saves OOF predictions (CSV) + summary (JSON)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score

RNG = 42

def parse_args():
    ap = argparse.ArgumentParser(description="Nested 10-fold validation for fraud detection.")
    ap.add_argument("--csv", required=True, help="Path to input CSV.")
    ap.add_argument("--target", required=True, help="Binary target column (0/1).")
    ap.add_argument("--output_dir", default="outputs_nested_cv", help="Where to save results.")
    ap.add_argument("--pca_components", type=int, default=None, help="Optional PCA n_components.")
    ap.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="none")
    ap.add_argument("--model", choices=["svc_rbf", "logreg"], default="svc_rbf")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--outer_folds", type=int, default=10)
    return ap.parse_args()

def build_pipeline(num_cols, pca_components, base_model):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RNG)))
    num_pipe = Pipeline(steps)
    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")
    if base_model == "svc_rbf":
        est = SVC(C=1.0, kernel="rbf", probability=True, random_state=RNG)
    else:
        est = LogisticRegression(solver="saga", max_iter=500, class_weight="balanced", random_state=RNG)
    return Pipeline([("pre", pre), ("clf", est)])

def tune_threshold(y_true, y_score, criterion="f1"):
    thr_grid = np.linspace(0.1, 0.9, 81)
    best_thr, best = 0.5, -np.inf
    for t in thr_grid:
        y_pred = (y_score >= t).astype(int)
        val = f1_score(y_true, y_pred, zero_division=0) if criterion=="f1" else balanced_accuracy_score(y_true, y_pred)
        if val > best:
            best, best_thr = val, t
    return float(best_thr), float(best)

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.target not in df.columns: raise ValueError(f"Target '{args.target}' not found.")
    y = df[args.target].astype(int).values
    X = df.drop(columns=[args.target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: raise ValueError("No numeric columns detected.")

    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=RNG)
    inner_cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=RNG)

    oof_scores = np.zeros(len(df)); oof_preds = np.zeros(len(df), dtype=int); oof_folds = np.zeros(len(df), dtype=int)
    chosen_thresholds, chosen_params = [], []

    grid = []
    if args.model == "svc_rbf":
        for C in [0.5, 1.0, 2.0]: grid.append({"model":"svc_rbf","C":C})
    else:
        for C in [0.5, 1.0, 2.0]: grid.append({"model":"logreg","C":C})

    for fold, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]; y_tr, y_te = y[tr], y[te]
        best_hp, best_val, best_thr = None, -np.inf, 0.5

        for hp in grid:
            vals, thrs = [], []
            for in_tr, in_va in inner_cv.split(X_tr, y_tr):
                X_in_tr, X_in_va = X_tr.iloc[in_tr], X_tr.iloc[in_va]
                y_in_tr, y_in_va = y_tr[in_tr], y_tr[in_va]

                pipe = build_pipeline(num_cols, args.pca_components, hp["model"])
                pipe.set_params(clf__C=hp["C"])

                if args.calibration in {"platt", "isotonic"}:
                    pre = pipe.named_steps["pre"]
                    est = pipe.named_steps["clf"]
                    est_cal = CalibratedClassifierCV(base_estimator=est,
                                                     method="sigmoid" if args.calibration=="platt" else "isotonic",
                                                     cv=3)
                    pipe = Pipeline([("pre", pre), ("clf", est_cal)])

                pipe.fit(X_in_tr, y_in_tr)
                y_score = pipe.predict_proba(X_in_va)[:, 1]
                thr, val = tune_threshold(y_in_va, y_score, criterion="f1")
                vals.append(val); thrs.append(thr)
            avg_val, avg_thr = float(np.mean(vals)), float(np.mean(thrs))
            if avg_val > best_val: best_val, best_thr, best_hp = avg_val, avg_thr, hp

        best = build_pipeline(num_cols, args.pca_components, best_hp["model"])
        best.set_params(clf__C=best_hp["C"])
        if args.calibration in {"platt", "isotonic"}:
            pre = best.named_steps["pre"]; est = best.named_steps["clf"]
            est_cal = CalibratedClassifierCV(base_estimator=est,
                                             method="sigmoid" if args.calibration=="platt" else "isotonic",
                                             cv=3)
            best = Pipeline([("pre", pre), ("clf", est_cal)])
        best.fit(X_tr, y_tr)
        y_score_te = best.predict_proba(X_te)[:, 1]
        y_pred_te = (y_score_te >= best_thr).astype(int)

        oof_scores[te] = y_score_te; oof_preds[te] = y_pred_te; oof_folds[te] = fold
        chosen_thresholds.append(best_thr); chosen_params.append(best_hp)
        print(f"[Fold {fold}] Best {best_hp} | Thr={best_thr:.3f}")

    oof = pd.DataFrame({"fold": oof_folds, "y_true": y, "y_score": oof_scores, "y_pred": oof_preds})
    oof.to_csv(outdir / "oof_predictions.csv", index=False)

    ba = balanced_accuracy_score(y, oof_preds)
    roc = roc_auc_score(y, oof_scores)
    pr = average_precision_score(y, oof_scores)
    f1 = f1_score(y, oof_preds)
    rec = recall_score(y, oof_preds)

    summary = {
        "balanced_accuracy": ba, "roc_auc": roc, "pr_auc": pr,
        "f1": f1, "recall": rec,
        "avg_threshold": float(np.mean(chosen_thresholds)),
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "calibration": args.calibration, "pca_components": args.pca_components,
        "chosen_params": chosen_params
    }
    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OOF Metrics ===")
    for k, v in summary.items():
        if isinstance(v, float): print(f"{k}: {v:.4f}")
        else: print(f"{k}: {v}")

if __name__ == "__main__":
    main()
