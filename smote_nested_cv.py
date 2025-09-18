#!/usr/bin/env python3
# -*- coding:: utf-8 -*-
"""
SMOTE + nested validation (leakage-safe).
- SMOTE applied *inside* training folds via imblearn Pipeline
- Outer: 10 stratified folds; Inner: 5 stratified folds
- Preprocessing: impute -> scale -> optional PCA
- Classifier: GradientBoosting (default) or LogisticRegression
- Metrics: BA, ROC-AUC, PR-AUC, Recall, F1
- Outputs: OOF predictions CSV + summary JSON
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RNG = 42

def parse_args():
    ap = argparse.ArgumentParser(description="SMOTE + nested validation (leakage-safe).")
    ap.add_argument("--csv", required=True, help="Path to input CSV.")
    ap.add_argument("--target", required=True, help="Binary target column (0/1).")
    ap.add_argument("--output_dir", default="outputs_smote_cv", help="Where to save results.")
    ap.add_argument("--pca_components", type=int, default=None, help="Optional PCA n_components.")
    ap.add_argument("--clf", choices=["gb", "logreg"], default="gb", help="Classifier choice.")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--outer_folds", type=int, default=10)
    ap.add_argument("--k_neighbors", type=int, default=5, help="SMOTE k_neighbors.")
    return ap.parse_args()

def build_pipeline(num_cols, pca_components, clf_name, smote_k):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RNG)))
    num_pipe = Pipeline(steps)
    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")
    if clf_name == "gb":
        clf = GradientBoostingClassifier(random_state=RNG)
    else:
        clf = LogisticRegression(solver="saga", max_iter=500, class_weight="balanced", random_state=RNG)
    return ImbPipeline(steps=[("pre", pre),
                              ("smote", SMOTE(k_neighbors=smote_k, random_state=RNG)),
                              ("clf", clf)])

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

    oof_score = np.zeros(len(df)); oof_pred = np.zeros(len(df), dtype=int); oof_fold = np.zeros(len(df), dtype=int)
    chosen_hps, chosen_thrs = [], []

    if args.clf == "gb":
        hp_grid = [{"clf":"gb","learning_rate":lr,"n_estimators":ne} for lr in [0.05,0.1] for ne in [100,200]]
    else:
        hp_grid = [{"clf":"logreg","C":c} for c in [0.5,1.0,2.0]]

    for fold, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]; y_tr, y_te = y[tr], y[te]
        best_hp, best_val, best_thr = None, -np.inf, 0.5

        for hp in hp_grid:
            vals, thrs = [], []
            for in_tr, in_va in inner_cv.split(X_tr, y_tr):
                X_in_tr, X_in_va = X_tr.iloc[in_tr], X_tr.iloc[in_va]
                y_in_tr, y_in_va = y_tr[in_tr], y_tr[in_va]

                pipe = build_pipeline(num_cols, args.pca_components, hp["clf"], args.k_neighbors)
                if hp["clf"] == "gb":
                    pipe.set_params(clf__learning_rate=hp["learning_rate"], clf__n_estimators=hp["n_estimators"])
                else:
                    pipe.set_params(clf__C=hp["C"])

                pipe.fit(X_in_tr, y_in_tr)
                y_score = pipe.predict_proba(X_in_va)[:, 1]
                thr, val = tune_threshold(y_in_va, y_score, criterion="f1")
                vals.append(val); thrs.append(thr)

            avg_val, avg_thr = float(np.mean(vals)), float(np.mean(thrs))
            if avg_val > best_val:
                best_val, best_thr, best_hp = avg_val, avg_thr, hp

        pipe = build_pipeline(num_cols, args.pca_components, best_hp["clf"], args.k_neighbors)
        if best_hp["clf"] == "gb":
            pipe.set_params(clf__learning_rate=best_hp["learning_rate"], clf__n_estimators=best_hp["n_estimators"])
        else:
            pipe.set_params(clf__C=best_hp["C"])
        pipe.fit(X_tr, y_tr)

        y_score_te = pipe.predict_proba(X_te)[:, 1]
        y_pred_te = (y_score_te >= best_thr).astype(int)

        oof_score[te] = y_score_te; oof_pred[te] = y_pred_te; oof_fold[te] = fold
        chosen_hps.append(best_hp); chosen_thrs.append(best_thr)
        print(f"[Fold {fold}] Best {best_hp} | Thr={best_thr:.3f}")

    oof = pd.DataFrame({"fold": oof_fold, "y_true": y, "y_score": oof_score, "y_pred": oof_pred})
    oof.to_csv(outdir / "oof_predictions_smote.csv", index=False)

    ba = balanced_accuracy_score(y, oof_pred)
    roc = roc_auc_score(y, oof_score)
    pr = average_precision_score(y, oof_score)
    f1 = f1_score(y, oof_pred)
    rec = recall_score(y, oof_pred)

    summary = {
        "balanced_accuracy": ba, "roc_auc": roc, "pr_auc": pr,
        "f1": f1, "recall": rec,
        "avg_threshold": float(np.mean(chosen_thrs)),
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "pca_components": args.pca_components, "smote_k_neighbors": args.k_neighbors,
        "chosen_hyperparams": chosen_hps
    }
    with open(outdir / "metrics_summary_smote.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OOF Metrics (SMOTE) ===")
    for k, v in summary.items():
        if isinstance(v, float): print(f"{k}: {v:.4f}")
        else: print(f"{k}: {v}")

if __name__ == "__main__":
    main()
