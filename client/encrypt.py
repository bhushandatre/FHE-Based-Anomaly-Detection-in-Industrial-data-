"""
client/encrypt.py
=================
Encrypts training and test data for the HE-SAD pipeline.

Training data → column-major format
    51 ciphertexts, one per feature
    ct_feature_j has all N_TRAIN row values in slots 0..N_TRAIN-1
    Saved to: encrypted/train/ct_feature_XX.seal

Test data → row-major format
    One ciphertext per row
    Each ct has 51 feature values in slots 0..50
    Saved to: encrypted/test/ct_XXXXXX.seal

Labels saved to: encrypted/test_labels.npy

Usage:
    python client/encrypt.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from seal import Ciphertext
from key import context, encoder, SCALE, load_client_keys

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR        = os.path.join(os.path.dirname(__file__), '..')
TRAIN_CSV       = os.path.join(BASE_DIR, "data", "normal_preprocessed.csv")
TEST_CSV        = os.path.join(BASE_DIR, "data", "swat_unified_dataset.csv")
TRAIN_OUT_DIR   = os.path.join(BASE_DIR, "encrypted", "train")
TEST_OUT_DIR    = os.path.join(BASE_DIR, "encrypted", "test")
LABELS_OUT      = os.path.join(BASE_DIR, "encrypted", "test_labels.npy")
N_TRAIN         = 5000
N_TEST          = 2000
N_FEATURES      = 51


def encrypt_train():
    """
    Loads training data (already normalized to [-1,1]),
    encrypts in column-major format.
    Saves 51 ciphertexts to encrypted/train/
    """
    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)

    encryptor, _, _, _ = load_client_keys()
    slot_count         = encoder.slot_count()

    print("[encrypt.py] Encrypting training data...")

    df      = pd.read_csv(TRAIN_CSV)
    X_train = df.values.astype(np.float64)[:N_TRAIN]

    print(f"  OK  Loaded {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"  OK  Value range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # Encrypt column-major: one ciphertext per feature
    for i in range(N_FEATURES):
        col     = X_train[:, i].tolist()
        padded  = col + [0.0] * (slot_count - len(col))
        plain   = encoder.encode(padded, SCALE)
        ct      = encryptor.encrypt(plain)
        ct_path = os.path.join(TRAIN_OUT_DIR, f"ct_feature_{i:02d}.seal")
        ct.save(ct_path)

        if (i + 1) % 10 == 0 or i == N_FEATURES - 1:
            print(f"  [{i+1:2d}/{N_FEATURES}] ct_feature_{i:02d}.seal saved")

    print(f"  OK  {N_FEATURES} training ciphertexts saved to {TRAIN_OUT_DIR}/")


def encrypt_test():
    """
    Loads test data from swat_unified_dataset.csv.
    Data is already normalized to [-1,1] by preprocess.py — NO renormalization.
    Encrypts row-major: one ciphertext per row.
    Saves 2000 ciphertexts to encrypted/test/
    Saves labels to encrypted/test_labels.npy
    """
    os.makedirs(TEST_OUT_DIR, exist_ok=True)

    encryptor, _, _, _ = load_client_keys()
    slot_count         = encoder.slot_count()

    print("[encrypt.py] Encrypting test data...")

    # Load unified dataset — already normalized by preprocess.py
    df_test = pd.read_csv(TEST_CSV)
    df_test.columns = df_test.columns.str.strip()

    y_test      = df_test['label'].values.astype(int)
    X_test_norm = df_test.drop(columns=['label']).values.astype(np.float64)

    n_normal = (y_test == 0).sum()
    n_attack = (y_test == 1).sum()

    print(f"  OK  {n_normal} normal + {n_attack} attack rows")
    print(f"  OK  Value range: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")

    # Save labels
    np.save(LABELS_OUT, y_test)
    print(f"  OK  Labels saved to {LABELS_OUT}")

    # Encrypt row-major: one ciphertext per test row
    for i in range(N_TEST):
        row     = X_test_norm[i].tolist()
        padded  = row + [0.0] * (slot_count - len(row))
        plain   = encoder.encode(padded, SCALE)
        ct      = encryptor.encrypt(plain)
        ct_path = os.path.join(TEST_OUT_DIR, f"ct_{i:06d}.seal")
        ct.save(ct_path)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1:4d}/{N_TEST}] rows encrypted")

    print(f"  OK  {N_TEST} test ciphertexts saved to {TEST_OUT_DIR}/")



if __name__ == "__main__":
    print("=" * 50)
    print("Client — Encrypt Data")
    print("=" * 50)
    encrypt_train()
    print()
    encrypt_test()
    print("\n  Encryption complete.")
    print(f"  Training : {TRAIN_OUT_DIR}/ ({N_FEATURES} ciphertexts)")
    print(f"  Test     : {TEST_OUT_DIR}/ ({N_TEST} ciphertexts)")
    print(f"  Labels   : {LABELS_OUT}")