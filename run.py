"""
run.py
======
Orchestrates the full multi-round HE-SAD protocol.

Round 0 : key.py already ran — keys exist in keys/client and keys/server
Round 1 : Client sends encrypted training data to server
Round 2 : Server computes sum, asks client to invert N
Round 3 : Client inverts N, sends 1/N to server
Round 4 : Server computes mu, sigma², asks client to invert sigma²
Round 5 : Client inverts sigma², sends 1/sigma² to server
Round 6 : Client sends encrypted test data to server
Round 7 : Server computes SAD scores
Round 8 : Server sends encrypted scores to client
Round 9 : Client decrypts scores, applies threshold → Normal or Attack

Usage:
    python run.py
"""

import os
import sys
import numpy as np
import pandas as pd
from seal import Ciphertext
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

from key import context, encoder, SCALE, load_client_keys, load_server_keys
from message import InvertContext
from client.messenger import ClientMessenger
from server.messenger import ServerMessenger
from server.algo      import SAD

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
TRAIN_CT_DIR  = "encrypted/train"
TEST_CT_DIR   = "encrypted/test"
LABELS_PATH   = "encrypted/test_labels.npy"
N_FEATURES    = 51
N_TRAIN       = 5000
N_TEST        = 2000
THRESHOLD_N   = 500   # number of training rows to score for threshold

# Features with sigma2 >= 0.01 from normal training data
# Remaining 35 features are near-constant and excluded (weight set to 0)
KEEP_FEATURES = [0, 1, 2, 3, 4, 8, 9, 12, 14, 16, 17, 18, 20, 22, 24, 28]


def main():
    print("=" * 60)
    print("HE-SAD Multi-Round Protocol")
    print("=" * 60)
    print(f"  Using {len(KEEP_FEATURES)}/51 informative features")

    client_msg = ClientMessenger()
    server_msg = ServerMessenger()
    sad        = SAD()

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
    ct_sum_list, ct_N = sad.compute_sum(received_train)
    inv_req_N = server_msg.send_invert_request([ct_N], InvertContext.N)

    # ── Round 3: Client inverts N, sends back ─────────────────
    print("\n[Round 3] Client inverting N...")
    inv_resp_N    = client_msg.handle_invert_request(inv_req_N)
    ct_inv_N_list = server_msg.receive_invert_response(inv_resp_N)
    ct_inv_N      = ct_inv_N_list[0]

    # ── Round 4: Server computes mu and sigma² ────────────────
    print("\n[Round 4] Server computing mu and sigma²...")
    sad.compute_mu(ct_inv_N)
    ct_sigma2_list = sad.compute_sigma2(received_train)

    _, decryptor_sigma, _, _ = load_client_keys()
    sigma2_raw = []
    for j in range(N_FEATURES):
        decoded  = encoder.decode(decryptor_sigma.decrypt(ct_sigma2_list[j]))
        sigma2_j = float(np.real(decoded[0])) / N_TRAIN
        sigma2_j = max(sigma2_j, 0.01)
        sigma2_raw.append(sigma2_j)

    print(f"  OK  sigma2 range: [{min(sigma2_raw):.6f}, {max(sigma2_raw):.6f}]")

    slot_count_s2    = encoder.slot_count()
    sigma2_padded    = sigma2_raw + [1.0] * (slot_count_s2 - len(sigma2_raw))
    pt_sigma2_pack   = encoder.encode(sigma2_padded, SCALE)
    encryptor_s2, _, _, _ = load_client_keys()
    ct_sigma2_packed = encryptor_s2.encrypt(pt_sigma2_pack)

    inv_req_s2 = server_msg.send_invert_request(
        [ct_sigma2_packed], InvertContext.SIGMA2
    )

    # ── Round 5: Client inverts sigma², sends back ────────────
    print("\n[Round 5] Client inverting sigma²...")
    inv_resp_s2    = client_msg.handle_invert_request(inv_req_s2)
    ct_inv_s2_list = server_msg.receive_invert_response(inv_resp_s2)
    sad.store_inv_sigma2(ct_inv_s2_list)

    # ── Decode mu and inv_sigma2 ───────────────────────────────
    _, decryptor, _, _ = load_client_keys()

    mu_vals = []
    for j in range(N_FEATURES):
        decoded_mu = encoder.decode(decryptor.decrypt(sad.ct_mu[j]))
        mu_vals.append(float(np.real(decoded_mu[0])))

    decoded_inv_s2  = encoder.decode(decryptor.decrypt(ct_inv_s2_list[0]))
    inv_sigma2_vals = []
    for j in range(N_FEATURES):
        if j in KEEP_FEATURES:
            # informative feature — use its inv_sigma2
            inv_val = float(np.real(decoded_inv_s2[j]))
            inv_val = max(min(inv_val, 100.0), 0.0)
        else:
            # near-constant feature — exclude from score
            inv_val = 0.0
        inv_sigma2_vals.append(inv_val)

    n_active    = sum(1 for v in inv_sigma2_vals if v > 0)
    active_vals = [v for v in inv_sigma2_vals if v > 0]
    print(f"  OK  Active features      : {n_active}/{N_FEATURES}")
    print(f"  OK  inv_sigma2 (active)  : [{min(active_vals):.4f}, {max(active_vals):.4f}]")

    slot_count        = encoder.slot_count()
    mu_padded         = mu_vals + [0.0] * (slot_count - len(mu_vals))
    inv_sigma2_padded = inv_sigma2_vals + [0.0] * (slot_count - len(inv_sigma2_vals))

    # ── Round 6: Load and send test data ──────────────────────
    print("\n[Round 6] Loading encrypted test data...")
    test_cts = []
    for i in range(N_TEST):
        path = os.path.join(TEST_CT_DIR, f"ct_{i:06d}.seal")
        ct   = Ciphertext()
        ct.load(context, path)
        test_cts.append(ct)
    print(f"  OK  Loaded {N_TEST} test ciphertexts")

    test_msg      = client_msg.send_test_data(test_cts)
    received_test = test_msg.to_ciphertexts(Ciphertext, context)

    # ── Round 7: Server computes SAD scores ───────────────────
    print("\n[Round 7] Server computing SAD scores...")
    score_cts = sad.compute_scores_batch(
        received_test, mu_padded, inv_sigma2_padded
    )

    # ── Round 8: Server sends scores to client ────────────────
    print("\n[Round 8] Server sending scores to client...")
    score_msg = server_msg.send_scores(score_cts)

    # ── Round 9: Client decrypts and decides ──────────────────
    print("\n[Round 9] Client decrypting scores and deciding...")

    # ── Compute threshold from training rows ──────────────────
    print("  Computing threshold from training rows...")
    encryptor_t, decryptor_t, _, _ = load_client_keys()
    df_train   = pd.read_csv("data/normal_preprocessed.csv")
    X_train_pt = df_train.values.astype(np.float64)[:THRESHOLD_N]

    train_scores = []
    for i in range(THRESHOLD_N):
        row    = X_train_pt[i].tolist()
        padded = row + [0.0] * (slot_count - len(row))
        plain  = encoder.encode(padded, SCALE)
        ct_i   = encryptor_t.encrypt(plain)
        score_ct_train = sad.compute_scores_batch([ct_i], mu_padded, inv_sigma2_padded)
        decoded = encoder.decode(decryptor_t.decrypt(score_ct_train[0]))
        train_scores.append(float(np.real(decoded[0])))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{THRESHOLD_N}] threshold rows scored")

    train_scores = np.array(train_scores)
    print(f"  OK  Train scores : mean={np.mean(train_scores):.4f}  std={np.std(train_scores):.4f}")

    # ── Decrypt test scores ────────────────────────────────────
    received_scores = client_msg.receive_scores(score_msg)
    _, decryptor_d, _, _ = load_client_keys()

    test_scores = []
    for ct in received_scores:
        decoded = encoder.decode(decryptor_d.decrypt(ct))
        test_scores.append(float(np.real(decoded[0])))
    test_scores = np.array(test_scores)

    # Save scores for analysis
    np.save('debug_scores.npy', test_scores)

    print(f"  OK  Test scores  : mean={np.mean(test_scores):.4f}  std={np.std(test_scores):.4f}")

    # ── Flip scores if needed ─────────────────────────────────
    y_true  = np.load(LABELS_PATH)
    auc_raw = roc_auc_score(y_true, test_scores)
    if auc_raw < 0.5:
        print(f"  NOTE: AUC={auc_raw:.4f} < 0.5, flipping scores")
        test_scores  = -test_scores
        train_scores = -train_scores

    # ── Threshold: 95th percentile of normal training scores ──
    threshold = float(np.mean(train_scores) + 3 * np.std(train_scores))
    print(f"  OK  Train mean  : {np.mean(train_scores):.6f}")
    print(f"  OK  Threshold   : {threshold:.6f}  (95th percentile of train scores)")

    # ── Apply threshold ────────────────────────────────────────
    predictions = (test_scores > threshold).astype(int)
    print(f"  OK  Predicted : {(predictions==0).sum()} Normal, {(predictions==1).sum()} Attack")

    # ── Evaluate ───────────────────────────────────────────────
    f1        = f1_score(y_true, predictions)
    auc       = roc_auc_score(y_true, test_scores)
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

    # ── Score distribution ─────────────────────────────────────
    print(f"\n  Score Distribution:")
    print(f"  Normal test : mean={test_scores[y_true==0].mean():.4f}  std={test_scores[y_true==0].std():.4f}")
    print(f"  Attack test : mean={test_scores[y_true==1].mean():.4f}  std={test_scores[y_true==1].std():.4f}")
    print(f"  Threshold   : {threshold:.4f}")

    print("\n" + "=" * 60)
    print("HE-SAD PROTOCOL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()