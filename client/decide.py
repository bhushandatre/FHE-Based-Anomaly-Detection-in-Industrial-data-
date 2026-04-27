"""
client/decide.py
================
Decrypts anomaly scores and decides Normal or Attack.
Only the client can do this — client holds the secret key.

Usage:
    from client.decide import compute_threshold, decide
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from key import context, encoder, load_client_keys
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score


def decrypt_scores(score_ciphertexts):
    """
    Decrypts list of score ciphertexts.
    Returns numpy array of scalar scores (one per test row).

    Args:
        score_ciphertexts : list of seal.Ciphertext

    Returns:
        numpy array of shape (n_rows,)
    """
    _, decryptor, _, _ = load_client_keys()

    scores = []
    for ct in score_ciphertexts:
        decoded = encoder.decode(decryptor.decrypt(ct))
        scores.append(float(np.real(decoded[0])))

    return np.array(scores)


def compute_threshold(train_score_ciphertexts):
    """
    Computes threshold from encrypted training scores.
    Threshold = mean + 3 * std of training scores.

    Args:
        train_score_ciphertexts : list of seal.Ciphertext (training row scores)

    Returns:
        float threshold value
    """
    train_scores = decrypt_scores(train_score_ciphertexts)
    threshold    = float(np.mean(train_scores) + 3 * np.std(train_scores))
    return threshold


def decide(score_ciphertexts, threshold, labels_path=None):
    """
    Decrypts scores, applies threshold, prints results.

    Args:
        score_ciphertexts : list of seal.Ciphertext
        threshold         : float threshold value
        labels_path       : path to test_labels.npy (optional, for evaluation)

    Returns:
        predictions : numpy array (0=Normal, 1=Attack)
        scores      : numpy array of decrypted scores
    """
    print("[decide.py] Decrypting scores...")

    scores = decrypt_scores(score_ciphertexts)

    # Auto flip if AUC < 0.5
    if labels_path and os.path.exists(labels_path):
        y_true = np.load(labels_path)
        auc    = roc_auc_score(y_true, scores)
        if auc < 0.5:
            scores    = -scores
            threshold = -threshold
            auc       = 1.0 - auc

    predictions = (scores > threshold).astype(int)

    print(f"  OK  Scores  : mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
    print(f"  OK  Threshold : {threshold:.4f}")
    print(f"  OK  Predicted : {(predictions==0).sum()} Normal, {(predictions==1).sum()} Attack")

    # Evaluate if labels available
    if labels_path and os.path.exists(labels_path):
        y_true    = np.load(labels_path)
        f1        = f1_score(y_true, predictions)
        auc       = roc_auc_score(y_true, scores)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall    = recall_score(y_true, predictions, zero_division=0)
        cm        = confusion_matrix(y_true, predictions)

        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-'*24}")
        print(f"  {'F1':<12} {f1:>10.4f}")
        print(f"  {'AUC':<12} {auc:>10.4f}")
        print(f"  {'Precision':<12} {precision:>10.4f}")
        print(f"  {'Recall':<12} {recall:>10.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return predictions, scores