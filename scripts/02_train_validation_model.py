"""
Script 02: Train validation models (Phase 1).
Works for any dataset via config file.

Usage (from protein_rna_ml/ folder):
    python scripts/02_train_validation_model.py --config configs/rbns_validation.yaml
    python scripts/02_train_validation_model.py --config configs/htr_selex_validation.yaml
"""

import argparse, json, os, time
import numpy as np
import yaml, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost not installed — skipped")


def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "auroc":       float(roc_auc_score(y_true, y_prob)),
        "auprc":       float(average_precision_score(y_true, y_prob)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name          = cfg["dataset"]["name"]
    processed_dir = cfg["dataset"]["processed_dir"]
    model_dir     = cfg["output"]["model_save_dir"]
    metrics_path  = cfg["output"]["metrics_path"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    print(f"\nLoading encoded data: {processed_dir}/")
    train = np.load(os.path.join(processed_dir, "train_kmer.npz"))
    val   = np.load(os.path.join(processed_dir, "val_kmer.npz"))
    X_tr, y_tr = train["X"], train["y"]
    X_va, y_va = val["X"],   val["y"]
    print(f"  Train {X_tr.shape}  {y_tr.mean():.1%} pos  |  Val {X_va.shape}  {y_va.mean():.1%} pos")

    mcfg = cfg["baseline_models"]
    results, best_auroc, best_name = {}, 0.0, None

    if mcfg["logistic_regression"]["enabled"]:
        t0 = time.time()
        print("\nLogistic Regression...")
        m = LogisticRegression(max_iter=mcfg["logistic_regression"].get("max_iter",1000),
                               C=mcfg["logistic_regression"].get("C",1.0), n_jobs=-1, random_state=42)
        m.fit(X_tr, y_tr)
        r = metrics(y_va, m.predict_proba(X_va)[:,1])
        r["t"] = round(time.time()-t0, 1)
        results["logistic_regression"] = r
        joblib.dump(m, os.path.join(model_dir,"logistic_regression.pkl"))
        print(f"  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  ({r['t']}s)")
        if r["auroc"] > best_auroc: best_auroc, best_name = r["auroc"], "logistic_regression"

    if mcfg["random_forest"]["enabled"]:
        t0 = time.time()
        print("\nRandom Forest...")
        rc = mcfg["random_forest"]
        m = RandomForestClassifier(n_estimators=rc.get("n_estimators",200),
                                   max_depth=rc.get("max_depth",15),
                                   n_jobs=-1, random_state=42)
        m.fit(X_tr, y_tr)
        r = metrics(y_va, m.predict_proba(X_va)[:,1])
        r["t"] = round(time.time()-t0, 1)
        results["random_forest"] = r
        joblib.dump(m, os.path.join(model_dir,"random_forest.pkl"))
        print(f"  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  ({r['t']}s)")
        if r["auroc"] > best_auroc: best_auroc, best_name = r["auroc"], "random_forest"

    if mcfg["xgboost"]["enabled"] and HAS_XGB:
        t0 = time.time()
        print("\nXGBoost...")
        xc = mcfg["xgboost"]
        m = XGBClassifier(n_estimators=xc.get("n_estimators",300), max_depth=xc.get("max_depth",6),
                          learning_rate=xc.get("learning_rate",0.1), subsample=xc.get("subsample",0.8),
                          colsample_bytree=xc.get("colsample_bytree",0.8),
                          eval_metric="logloss", n_jobs=-1, random_state=42, verbosity=1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)
        r = metrics(y_va, m.predict_proba(X_va)[:,1])
        r["t"] = round(time.time()-t0, 1)
        results["xgboost"] = r
        joblib.dump(m, os.path.join(model_dir,"xgboost.pkl"))
        print(f"  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  ({r['t']}s)")
        if r["auroc"] > best_auroc: best_auroc, best_name = r["auroc"], "xgboost"

    print(f"\n{'─'*55}")
    print(f"  {'Model':<25} {'AUROC':>7} {'AUPRC':>7} {'Acc':>7} {'F1':>7}")
    print(f"  {'─'*53}")
    for mn, r in results.items():
        print(f"  {mn:<25} {r['auroc']:>7.4f} {r['auprc']:>7.4f} {r['accuracy']:>7.4f} {r['f1']:>7.4f}")
    print(f"\n  Best: {best_name}  AUROC={best_auroc:.4f}")

    pt, wt = cfg["thresholds"]["auroc_pass"], cfg["thresholds"]["auroc_warn"]
    status = "PASS" if best_auroc >= pt else ("WARN" if best_auroc >= wt else "FAIL")
    msgs = {"PASS": f"✅ PASS — genuine signal. Proceed to 03_evaluate_validation.py",
            "WARN": f"⚠️  WARN — marginal. Investigate per-protein performance.",
            "FAIL": f"❌ FAIL — no signal. Do NOT add to Phase 2."}
    print(f"\n  {msgs[status]}")

    summary = {"dataset": name, "best_model": best_name, "best_val_auroc": best_auroc,
               "status": status, "models": results}
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {metrics_path}")
    print(f"  Models → {model_dir}/")

if __name__ == "__main__":
    main()
