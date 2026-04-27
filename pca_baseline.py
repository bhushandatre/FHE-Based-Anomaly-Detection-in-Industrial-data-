"""
PCA Based Anomaly Detector - Baseline
=======================================
Unsupervised anomaly detection using PCA reconstruction error.

Trains on normal data only, tests on mixed normal+attack data.

Algorithm:
    Training : fit PCA on normal data, choose k (90% variance)
    Scoring  : reconstruction error = ||x - PCA_reconstruct(x)||^2
    Threshold: mean + 3*std of training reconstruction errors

Corrections vs original:
    1. Removed StandardScaler — HE pipeline does NOT use StandardScaler,
       it works directly on [-1,1] normalized data. Using StandardScaler
       made PCA baseline incomparable to HE-PCA.
    2. Test normalization uses min/max from ALL normal rows in raw
       unified dataset — not from already-normalized X_train
       (avoids double normalization, matches HE pipeline)

Usage:
    python pca_baseline.py
"""

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════

DATASET_NAME    = "SWaT"
TRAIN_FILE      = "data/normal_preprocessed.csv"   # already normalized [-1,1]
TEST_FILE       = "data/swat_unified_dataset.csv"  # raw values with labels
DROP_COLS       = ['Timestamp', 'Normal/Attack', 'label']
LABEL_COL       = 'label'
OUTPUT_DIR      = "baseline_results"
VARIANCE_THRESH = 0.90    # keep components explaining 90% variance
N_TRAIN         = 5000
N_TEST          = 2000
ATTACK_RATIO    = 0.10    # 10% attack = 200 attack, 1800 normal
SEED            = 42

# ═══════════════════════════════════════════════

import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"PCA Anomaly Detector — {DATASET_NAME}")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: LOAD TRAINING DATA
# normal_preprocessed.csv is already normalized to [-1,1]
# NO StandardScaler — HE pipeline works on raw [-1,1] data directly
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading training data...")

df_train = pd.read_csv(TRAIN_FILE)
df_train.columns = df_train.columns.str.strip()
df_train = df_train.drop(
    columns=[c for c in DROP_COLS if c in df_train.columns],
    errors='ignore'
)
X_train = df_train.values.astype(np.float64)[:N_TRAIN]

print(f"  OK  {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
print(f"  OK  Range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"  OK  No StandardScaler — matches HE pipeline")

# ─────────────────────────────────────────────
# STEP 2: FIT PCA WITH OPTIMAL k
# ─────────────────────────────────────────────
print("\n[STEP 2] Fitting PCA...")

t_start  = time.time()
pca_full = PCA()
pca_full.fit(X_train)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
k      = int(np.argmax(cumvar >= VARIANCE_THRESH) + 1)

print(f"  OK  Optimal k: {k} components ({cumvar[k-1]*100:.2f}% variance explained)")
print(f"  OK  Variance threshold: {VARIANCE_THRESH*100:.0f}%")
print(f"  OK  Top 5 components:")
for i in range(min(5, k)):
    print(f"      PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}%  "
          f"(cumulative: {cumvar[i]*100:.2f}%)")

pca        = PCA(n_components=k)
pca.fit(X_train)
train_time = time.time() - t_start
print(f"  OK  Fitted in {train_time:.3f}sec")

# ─────────────────────────────────────────────
# STEP 3: COMPUTE THRESHOLD ON TRAINING DATA
# ─────────────────────────────────────────────
print("\n[STEP 3] Computing threshold...")

def recon_error(X, pca_model):
    """PCA reconstruction error = ||x - V^T V (x - mu)||^2"""
    X_proj  = pca_model.transform(X)
    X_recon = pca_model.inverse_transform(X_proj)
    return np.sum((X - X_recon) ** 2, axis=1)

train_errors = recon_error(X_train, pca)
threshold    = np.mean(train_errors) + 3 * np.std(train_errors)

print(f"  OK  Train error mean : {np.mean(train_errors):.4f}")
print(f"  OK  Train error std  : {np.std(train_errors):.4f}")
print(f"  OK  Threshold        : {threshold:.4f}")

# ─────────────────────────────────────────────
# STEP 4: LOAD AND SAMPLE TEST DATA
# ─────────────────────────────────────────────
print("\n[STEP 4] Loading and sampling test data...")

df_full = pd.read_csv(TEST_FILE)
df_full.columns = df_full.columns.str.strip()

n_attack = int(N_TEST * ATTACK_RATIO)
n_normal = N_TEST - n_attack

df_test = pd.concat([
    df_full[df_full[LABEL_COL] == 0].sample(n=n_normal, random_state=SEED),
    df_full[df_full[LABEL_COL] == 1].sample(n=n_attack, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

y_test     = df_test[LABEL_COL].values.astype(int)
X_test_raw = df_test.drop(
    columns=[c for c in DROP_COLS if c in df_test.columns],
    errors='ignore'
).values.astype(np.float64)

print(f"  OK  {len(y_test):,} rows — Normal: {(y_test==0).sum():,}  Attack: {(y_test==1).sum():,}")
print(f"  OK  Raw test range: [{X_test_raw.min():.4f}, {X_test_raw.max():.4f}]")

assert X_train.shape[1] == X_test_raw.shape[1], \
    f"Feature mismatch: Train={X_train.shape[1]} Test={X_test_raw.shape[1]}"

# ─────────────────────────────────────────────
# STEP 5: NORMALIZE TEST DATA
# Use min/max from ALL normal rows in raw unified dataset
# (not from already-normalized X_train — avoids double normalization)
# This mirrors exactly what encrypt.py does
# ─────────────────────────────────────────────
print("\n[STEP 5] Normalizing test data...")

X_normal_raw = df_full[df_full[LABEL_COL] == 0].drop(
    columns=[c for c in DROP_COLS if c in df_full.columns],
    errors='ignore'
).values.astype(np.float64)

feat_min = X_normal_raw.min(axis=0)
feat_max = X_normal_raw.max(axis=0)
feat_rng = np.where(feat_max - feat_min == 0, 1.0, feat_max - feat_min)

X_test_norm = np.clip(
    2.0 * (X_test_raw - feat_min) / feat_rng - 1.0,
    -1.0, 1.0
)

print(f"  OK  Normalized test range: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")

# ─────────────────────────────────────────────
# STEP 6: SCORE TEST DATA
# ─────────────────────────────────────────────
print("\n[STEP 6] Scoring test data...")

t_start     = time.time()
test_errors = recon_error(X_test_norm, pca)
y_pred      = (test_errors > threshold).astype(int)
test_time   = time.time() - t_start

print(f"  OK  Normal mean error : {test_errors[y_test==0].mean():.4f}")
print(f"  OK  Attack mean error : {test_errors[y_test==1].mean():.4f}")
print(f"  OK  Scored in         : {test_time:.4f}sec")

# ─────────────────────────────────────────────
# STEP 7: EVALUATE
# ─────────────────────────────────────────────
print("\n[STEP 7] Evaluating...")

f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
auc       = roc_auc_score(y_test, test_errors)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  {'Metric':<12} {'Value':>10}")
print(f"  {'-'*24}")
print(f"  {'F1':<12} {f1:>10.4f}")
print(f"  {'AUC':<12} {auc:>10.4f}")
print(f"  {'Precision':<12} {precision:>10.4f}")
print(f"  {'Recall':<12} {recall:>10.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# ─────────────────────────────────────────────
# STEP 8: PLOT
# ─────────────────────────────────────────────
print("\n[STEP 8] Saving plot...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'PCA Baseline — {DATASET_NAME}', fontsize=14, fontweight='bold')

ax = axes[0]
p99 = np.percentile(test_errors, 99)
ax.hist(test_errors[y_test==0][test_errors[y_test==0] <= p99],
        bins=60, alpha=0.6, color='green', label='Normal', density=True)
ax.hist(test_errors[y_test==1][test_errors[y_test==1] <= p99],
        bins=60, alpha=0.6, color='red',   label='Attack', density=True)
ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
           label=f'Threshold={threshold:.2f}')
ax.set_xlabel('Reconstruction Error'); ax.set_ylabel('Density')
ax.set_title(f'Error Distribution (k={k})'); ax.legend()

ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, test_errors)
ax.plot(fpr, tpr, color='green', lw=2, label=f'PCA (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
ax.set_title(f'Confusion Matrix\nF1={f1:.3f}')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'pca_baseline_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK  Plot saved: {plot_path}")

# ─────────────────────────────────────────────
# STEP 9: SAVE RESULTS
# ─────────────────────────────────────────────
results = {
    'model'   : 'Plaintext-PCA',
    'dataset' : DATASET_NAME,
    'train_config': {
        'train_rows'        : int(X_train.shape[0]),
        'test_rows'         : int(X_test_raw.shape[0]),
        'normal_in_test'    : int((y_test==0).sum()),
        'attack_in_test'    : int((y_test==1).sum()),
        'features'          : int(X_train.shape[1]),
        'k_components'      : int(k),
        'variance_explained': round(float(cumvar[k-1]), 4),
        'variance_threshold': VARIANCE_THRESH,
        'standardscaler'    : False,
    },
    'metrics': {
        'f1'        : round(float(f1),        4),
        'roc_auc'   : round(float(auc),       4),
        'precision' : round(float(precision), 4),
        'recall'    : round(float(recall),    4),
        'threshold' : round(float(threshold), 4),
    },
    'confusion_matrix': {
        'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
        'FN': int(cm[1,0]), 'TP': int(cm[1,1])
    },
    'scores': test_errors.tolist(),
    'labels': y_test.tolist()
}

results_path = os.path.join(OUTPUT_DIR, 'pca_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  OK  Results saved: {results_path}")

print("\n" + "=" * 60)
print(f"PCA BASELINE COMPLETE — {DATASET_NAME}")
print("=" * 60)
print(f"  Train rows : {X_train.shape[0]:,}")
print(f"  Test rows  : {X_test_raw.shape[0]:,}")
print(f"  Components : {k} ({cumvar[k-1]*100:.2f}% variance)")
print(f"  F1         : {f1:.4f}")
print(f"  AUC        : {auc:.4f}")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f}")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  Results saved to {OUTPUT_DIR}/")