"""
Script 02b: Lightweight Phase 1 validation for HTR-SELEX PRJEB25907.
Encodes k-mers on-the-fly from TSV splits (no giant NPZ needed).
Saves: models/saved/htr_selex_validation/{lr,rf,xgb}.pkl
       results/htr_selex/metrics/validation_results.json
       results/htr_selex/metrics/test_evaluation.json
       results/htr_selex/plots/roc_pr_curves.png
       results/htr_selex/plots/per_protein_auroc.png

Usage (from protein_rna_ml/):
    python scripts/02b_train_htr_selex_validation.py
"""

import json, os, sys, time
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── k-mer encoding ────────────────────────────────────────────────────────────
RNA_ALPHA = "AUGC"
AA_ALPHA  = "ACDEFGHIKLMNPQRSTVWY"

def build_kmer_index(alphabet, k):
    return {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}

RNA_IDX  = build_kmer_index(RNA_ALPHA, 4)   # 256 features
PROT_IDX = build_kmer_index(AA_ALPHA,  3)   # 8000 features
N_RNA, N_PROT = len(RNA_IDX), len(PROT_IDX)

def encode_seq(seq, idx, k):
    vec = np.zeros(len(idx), dtype=np.float32)
    seq = str(seq).upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in idx:
            vec[idx[kmer]] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec

def encode_df(df, rna_col="rna_sequence", prot_col="protein_sequence"):
    X = np.zeros((len(df), N_RNA + N_PROT), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        X[i, :N_RNA]  = encode_seq(getattr(row, rna_col),  RNA_IDX,  4)
        X[i, N_RNA:]  = encode_seq(getattr(row, prot_col), PROT_IDX, 3)
        if i % 10000 == 0:
            print(f"    encoded {i}/{len(df)}", end="\r")
    print()
    return X

# ── paths ─────────────────────────────────────────────────────────────────────
SPLIT_DIR  = "data/splits/htr_selex"
METRICS_DIR = "results/htr_selex/metrics"
PLOTS_DIR   = "results/htr_selex/plots"
MODEL_DIR   = "models/saved/htr_selex_validation"
PASS_THRESH = 0.70

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ── load splits ───────────────────────────────────────────────────────────────
print("\n=== Phase 1 Validation: HTR-SELEX PRJEB25907 ===")
print(f"\nLoading splits from {SPLIT_DIR}/")
train_df = pd.read_csv(f"{SPLIT_DIR}/train.tsv", sep="\t")
val_df   = pd.read_csv(f"{SPLIT_DIR}/val.tsv",   sep="\t")
test_df  = pd.read_csv(f"{SPLIT_DIR}/test.tsv",  sep="\t")
print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
print(f"  Proteins: {train_df['protein_name'].nunique()} train  "
      f"{val_df['protein_name'].nunique()} val  "
      f"{test_df['protein_name'].nunique()} test")

# ── encode ────────────────────────────────────────────────────────────────────
print("\nEncoding train k-mers...")
X_tr = encode_df(train_df); y_tr = train_df["binding_label"].values
print("Encoding val k-mers...")
X_val = encode_df(val_df);  y_val = val_df["binding_label"].values
print("Encoding test k-mers...")
X_te = encode_df(test_df);  y_te = test_df["binding_label"].values

scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te  = scaler.transform(X_te)

# ── train models ──────────────────────────────────────────────────────────────
def fit_eval(name, model, X_tr, y_tr, X_val, y_val):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t = round(time.time() - t0, 1)
    prob = model.predict_proba(X_val)[:, 1]
    pred = model.predict(X_val)
    return {
        "auroc": float(roc_auc_score(y_val, prob)),
        "auprc": float(average_precision_score(y_val, prob)),
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_val, prob)),
        "t": t,
    }, model

models_cfg = {
    "logistic_regression": LogisticRegression(
        max_iter=300, C=1.0, solver="saga", n_jobs=-1),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=42),
    "xgboost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=20,
        n_jobs=-1, random_state=42, verbosity=0),
}

results = {}
for mname, clf in models_cfg.items():
    print(f"\nTraining {mname}...")
    if mname == "xgboost":
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        prob = clf.predict_proba(X_val)[:, 1]
        pred = clf.predict(X_val)
        t = 0
        metrics = {
            "auroc": float(roc_auc_score(y_val, prob)),
            "auprc": float(average_precision_score(y_val, prob)),
            "accuracy": float(accuracy_score(y_val, pred)),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_val, prob)),
            "t": t,
        }
    else:
        metrics, clf = fit_eval(mname, clf, X_tr, y_tr, X_val, y_val)
    results[mname] = metrics
    joblib.dump(clf, f"{MODEL_DIR}/{mname}.pkl")
    print(f"  Val AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}  t={metrics['t']}s")

best_model = max(results, key=lambda m: results[m]["auroc"])
best_auroc = results[best_model]["auroc"]
status = "PASS" if best_auroc >= PASS_THRESH else ("WARN" if best_auroc >= 0.65 else "FAIL")

val_out = {
    "dataset": "htr_selex_prjeb25907",
    "best_model": best_model,
    "best_val_auroc": best_auroc,
    "status": status,
    "models": results,
}
with open(f"{METRICS_DIR}/validation_results.json", "w") as f:
    json.dump(val_out, f, indent=2)
print(f"\n{'✅' if status=='PASS' else '⚠️'} Status: {status}  Best: {best_model} AUROC={best_auroc:.4f}")

# ── test evaluation (best model) ──────────────────────────────────────────────
print("\n=== Test evaluation ===")
best_clf = joblib.load(f"{MODEL_DIR}/{best_model}.pkl")
prob_te  = best_clf.predict_proba(X_te)[:, 1]
pred_te  = best_clf.predict(X_te)

overall = {
    "auroc":    float(roc_auc_score(y_te, prob_te)),
    "auprc":    float(average_precision_score(y_te, prob_te)),
    "accuracy": float(accuracy_score(y_te, pred_te)),
    "f1":       float(f1_score(y_te, pred_te, zero_division=0)),
    "brier":    float(brier_score_loss(y_te, prob_te)),
}
print(f"  Overall  AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")

test_df = test_df.copy()
test_df["prob"] = prob_te
per_protein = []
flagged = []
for prot, grp in test_df.groupby("protein_name"):
    if grp["binding_label"].nunique() < 2:
        continue
    auroc = float(roc_auc_score(grp["binding_label"], grp["prob"]))
    auprc = float(average_precision_score(grp["binding_label"], grp["prob"]))
    entry = {"protein": prot, "auroc": auroc, "auprc": auprc,
             "n": int(len(grp)), "note": "ok"}
    per_protein.append(entry)
    if auroc < PASS_THRESH:
        flagged.append({"protein": prot, "auroc": auroc, "n": int(len(grp))})

per_protein.sort(key=lambda x: -x["auroc"])
pp_aurocs = [p["auroc"] for p in per_protein]
pp_summary = {
    "median": float(np.median(pp_aurocs)),
    "mean":   float(np.mean(pp_aurocs)),
    "min":    float(np.min(pp_aurocs)),
    "max":    float(np.max(pp_aurocs)),
    "n_flagged": len(flagged),
}
print(f"  Per-protein  median={pp_summary['median']:.4f}  "
      f"min={pp_summary['min']:.4f}  flagged={len(flagged)}")

te_status = "PASS" if overall["auroc"] >= PASS_THRESH else "WARN"
test_out = {
    "dataset": "htr_selex_prjeb25907",
    "model": best_model,
    "overall": overall,
    "per_protein_summary": pp_summary,
    "per_protein": per_protein,
    "flagged": flagged,
    "status": te_status,
    "message": "✅ Ready for Phase 2" if te_status=="PASS" else "⚠️  Marginal — investigate before Phase 2",
}
with open(f"{METRICS_DIR}/test_evaluation.json", "w") as f:
    json.dump(test_out, f, indent=2)

# ── plots ─────────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve, precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fpr, tpr, _ = roc_curve(y_te, prob_te)
axes[0].plot(fpr, tpr, lw=2, label=f"AUC={overall['auroc']:.3f}")
axes[0].plot([0,1],[0,1],"k--"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC — HTR-SELEX PRJEB25907 test"); axes[0].legend()
prec, rec, _ = precision_recall_curve(y_te, prob_te)
axes[1].plot(rec, prec, lw=2, label=f"AUPRC={overall['auprc']:.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("PR — HTR-SELEX PRJEB25907 test"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{PLOTS_DIR}/roc_pr_curves.png", dpi=150); plt.close()

prots_sorted = sorted(per_protein, key=lambda x: x["auroc"])
colors = ["crimson" if p["auroc"] < PASS_THRESH else "steelblue" for p in prots_sorted]
fig, ax = plt.subplots(figsize=(14, max(6, len(prots_sorted)*0.25)))
ax.barh([p["protein"] for p in prots_sorted], [p["auroc"] for p in prots_sorted], color=colors)
ax.axvline(PASS_THRESH, color="red", linestyle="--", label=f"threshold={PASS_THRESH}")
ax.set_xlabel("AUROC"); ax.set_title("Per-protein AUROC — HTR-SELEX PRJEB25907 test")
ax.legend(); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/per_protein_auroc.png", dpi=150); plt.close()

print(f"\n✅ HTR-SELEX PRJEB25907 Phase 1 validation COMPLETE ({status})")
print(f"   Results: {METRICS_DIR}/")
print(f"   Plots:   {PLOTS_DIR}/")
