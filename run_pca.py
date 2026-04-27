"""
run_pca.py
==========
Orchestrates the full multi-round HE-PCA protocol.

PCA Anomaly Detection:
    score = || (x - mu) - V.T @ V @ (x - mu) ||²
    High score = point not well represented by normal subspace = anomaly

Rounds:
    Round 1 : Client sends encrypted training data
    Round 2 : Server computes sum, asks client to invert N
    Round 3 : Client inverts N, sends back
    Round 4 : Server computes mu, fits PCA on plaintext
    Round 5 : Client sends encrypted test data
    Round 6 : Server computes PCA reconstruction error scores
    Round 7 : Server sends encrypted scores to client
    Round 8 : Client decrypts scores, applies threshold → Normal or Attack

Note: PCA does not need sigma² inversion — no Round 5 inversion needed.
      Division only needed for mu (1/N).

Usage:
    python run_pca.py
"""

import os
import sys
import numpy as np
import pandas as pd
from seal import Ciphertext
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix
)

from key import context, encoder, SCALE, load_client_keys, load_server_keys
from message import InvertContext
from client.messenger import ClientMessenger
from server.messenger import ServerMessenger
from server.pca_algo  import PCA_HE
from client.decide    import decide

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_CT_DIR  = "encrypted/train"
TEST_CT_DIR   = "encrypted/test"
LABELS_PATH   = "encrypted/test_labels.npy"
TRAIN_CSV     = "data/normal_preprocessed.csv"
N_FEATURES    = 51
N_TRAIN       = 5000
N_TEST        = 2000
N_COMPONENTS  = 10
THRESHOLD_N   = 500


def main():
    print("=" * 60)
    print("HE-PCA Multi-Round Protocol")
    print(f"  n_components = {N_COMPONENTS}")
    print("=" * 60)

    client_msg = ClientMessenger()
    server_msg = ServerMessenger()
    pca_he     = PCA_HE(n_components=N_COMPONENTS)

    # ── Round 1: Load and send training data ──────────────────
    print("\n[Round 1] Loading encrypted training data...")
    train_cts = []
    for j in range(N_FEATURES):
        path = os.path.join(TRAIN_CT_DIR, f"ct_feature_{j:02d}.seal")
        ct   = Ciphertext()
        ct.load(context, path)
        train_cts.append(ct)
    print(f"  OK  Loaded {N_FEATURES} training ciphertexts")

    train_msg      = client_msg.send_train_data(train_cts)
    received_train = train_msg.to_ciphertexts(Ciphertext, context)

    # ── Round 2: Server computes sum, requests 1/N ────────────
    print("\n[Round 2] Server computing feature sums...")
    ct_sum_list, ct_N = pca_he.compute_sum(received_train)
    inv_req_N         = server_msg.send_invert_request([ct_N], InvertContext.N)

    # ── Round 3: Client inverts N, sends back ─────────────────
    print("\n[Round 3] Client inverting N...")
    inv_resp_N    = client_msg.handle_invert_request(inv_req_N)
    ct_inv_N_list = server_msg.receive_invert_response(inv_resp_N)
    ct_inv_N      = ct_inv_N_list[0]

    # ── Round 4: Server computes mu, fits PCA ─────────────────
    print("\n[Round 4] Server computing mu and fitting PCA...")
    pca_he.compute_mu(ct_inv_N)

    # Compute plaintext mu and fit PCA
    df_train    = pd.read_csv(TRAIN_CSV)
    X_train_pt  = df_train.values.astype(np.float64)[:N_TRAIN]
    mu_plaintext = X_train_pt.mean(axis=0)

    V = pca_he.fit_pca(TRAIN_CSV, mu_plaintext)

    # ── Round 5: Load and send test data ──────────────────────
    print("\n[Round 5] Loading encrypted test data...")
    test_cts = []
    for i in range(N_TEST):
        path = os.path.join(TEST_CT_DIR, f"ct_{i:06d}.seal")
        ct   = Ciphertext()
        ct.load(context, path)
        test_cts.append(ct)
    print(f"  OK  Loaded {N_TEST} test ciphertexts")

    test_msg      = client_msg.send_test_data(test_cts)
    received_test = test_msg.to_ciphertexts(Ciphertext, context)

    # ── Round 6: Server computes PCA scores ───────────────────
    print("\n[Round 6] Server computing PCA reconstruction error scores...")
    score_cts = pca_he.compute_scores_batch(received_test, mu_plaintext, V)

    # ── Round 7: Server sends scores to client ────────────────
    print("\n[Round 7] Server sending scores to client...")
    score_msg = server_msg.send_scores(score_cts)

    # ── Round 8: Client decrypts and decides ──────────────────
    print("\n[Round 8] Client decrypting scores and deciding...")

    # Compute threshold from training rows
    print("  Computing threshold from training rows...")
    encryptor_t, decryptor_t, _, _ = load_client_keys()
    slot_count = encoder.slot_count()

    pt_mu_padded = mu_plaintext.tolist() + [0.0] * (slot_count - len(mu_plaintext))
    pt_V_rows    = []
    for k in range(N_COMPONENTS):
        v_row = V[k].tolist() + [0.0] * (slot_count - len(V[k]))
        pt_V_rows.append(v_row)

    train_scores = []
    for i in range(THRESHOLD_N):
        row    = X_train_pt[i].tolist()
        padded = row + [0.0] * (slot_count - len(row))
        plain  = encoder.encode(padded, SCALE)
        ct_i   = encryptor_t.encrypt(plain)

        score_ct = pca_he.compute_score(ct_i, pt_mu_padded, pt_V_rows)
        decoded  = encoder.decode(decryptor_t.decrypt(score_ct))
        train_scores.append(float(np.real(decoded[0])))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{THRESHOLD_N}] threshold rows scored")

    train_scores = np.array(train_scores)
    threshold    = float(np.mean(train_scores) + 3 * np.std(train_scores))
    print(f"  OK  Train mean : {np.mean(train_scores):.6f}")
    print(f"  OK  Train std  : {np.std(train_scores):.6f}")
    print(f"  OK  Threshold  : {threshold:.6f}")

    # Decrypt and decide
    received_scores = client_msg.receive_scores(score_msg)

    predictions, scores = decide(
        received_scores,
        threshold,
        labels_path=LABELS_PATH
    )

    # ── Plaintext comparison ───────────────────────────────────
    print("\n  Plaintext PCA comparison:")
    X_centered  = X_train_pt - mu_plaintext
    X_proj      = X_centered @ V.T
    X_recon     = X_proj @ V
    train_err_pt = np.sum((X_centered - X_recon) ** 2, axis=1)
    threshold_pt = np.mean(train_err_pt) + 3 * np.std(train_err_pt)

    df_test = pd.read_csv("data/swat_unified_dataset.csv")
    df_test.columns = df_test.columns.str.strip()
    y_true   = np.load(LABELS_PATH)

    print(f"  Plaintext threshold : {threshold_pt:.6f}")
    print(f"  HE threshold        : {threshold:.6f}")

    print("\n" + "=" * 60)
    print("HE-PCA PROTOCOL COMPLETE")
    print("=" * 60)
    print(f"  n_components : {N_COMPONENTS}")
    print(f"  Train rows   : {N_TRAIN}")
    print(f"  Test rows    : {N_TEST}")
    print(f"  Threshold    : {threshold:.6f}")


if __name__ == "__main__":
    main()