"""
server/pca_algo.py
==================
PCA anomaly detection running on encrypted data.
Server never decrypts anything except via multi-round protocol.

PCA Logic:
    Training:
        1. Compute mu from encrypted training data (column-major)
        2. Fit PCA components V on plaintext training data
           (V is a global model parameter — acceptable to compute in plaintext)

    Inference:
        For each encrypted test row ct_x:
        1. ct_xc   = ct_x - pt_mu               (center)
        2. ct_proj_k = dot(ct_xc, pt_v_k)       (project onto k-th component)
                     = multiply_plain + slot_sum
        3. ct_recon  = sum_k [ct_proj_k * pt_v_k] (reconstruct)
        4. ct_resid  = ct_xc - ct_recon          (residual)
        5. ct_score  = sum(ct_resid²)            (reconstruction error)

    Depth budget (5 usable levels):
        sub_plain (center)         : 0 levels
        multiply_plain (project)   : 1 level
        slot_sum (projection)      : 0 levels
        multiply_plain (reconstruct): 1 level
        sub (residual)             : 0 levels
        square (error)             : 1 level
        slot_sum (sum error)       : 0 levels
        Total                      : 3 levels ✅

Division:
    mu is computed via multi-round protocol (same as SAD):
        Server sends ct_N → client inverts → server gets ct_inv_N
        ct_mu = ct_sum * ct_inv_N

Usage:
    from server.pca_algo import PCA_HE
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from seal import Ciphertext
from key import context, encoder, evaluator, SCALE, load_server_keys

# Rotation steps for slot_sum of N_TRAIN=5000 training slots
ROTATION_STEPS_TRAIN = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Rotation steps for slot_sum of 51 feature slots
ROTATION_STEPS_FEAT  = [1, 2, 4, 8, 16, 32]

N_TRAIN    = 5000
N_FEATURES = 51


class PCA_HE:
    def __init__(self, n_components=10):
        self.encryptor, self.relin_keys, self.galois_keys = load_server_keys()
        self.slot_count  = encoder.slot_count()
        self.n_components = n_components

        # Model parameters
        self.ct_sum  = None   # list of 51 encrypted feature sums
        self.ct_mu   = None   # list of 51 encrypted means
        self.V       = None   # PCA components (n_components x n_features) — plaintext
        self.mu_pt   = None   # plaintext mu vector (51 values)

    # ─────────────────────────────────────────
    # HELPER: slot_sum
    # Sums N slots using binary rotation tree
    # ─────────────────────────────────────────
    def slot_sum(self, ct, rotation_steps, max_slots=None):
        """Sum slots using rotations. Returns ct where every slot = sum."""
        ct_result = Ciphertext(ct)
        total     = 1
        for step in rotation_steps:
            if max_slots and total >= max_slots:
                break
            ct_rot = evaluator.rotate_vector(ct_result, step, self.galois_keys)
            evaluator.mod_switch_to_inplace(ct_rot, ct_result.parms_id())
            ct_rot.scale(ct_result.scale())
            evaluator.add_inplace(ct_result, ct_rot)
            total += step
        return ct_result

    # ─────────────────────────────────────────
    # STEP 1: Compute sum per feature
    # Same as SAD — returns ct_sum and ct_N for client to invert
    # ─────────────────────────────────────────
    def compute_sum(self, train_ciphertexts):
        """
        Computes slot_sum for each of 51 training ciphertexts.
        Returns ct_sum list and ct_N for client to invert.

        Args:
            train_ciphertexts : list of 51 seal.Ciphertext (column-major)

        Returns:
            ct_sum : list of 51 ciphertexts
            ct_N   : encrypted N for client to invert
        """
        print("[pca_algo.py] Computing feature sums...")

        self.ct_sum = []
        for j in range(N_FEATURES):
            ct_sum_j = self.slot_sum(
                train_ciphertexts[j],
                ROTATION_STEPS_TRAIN,
                max_slots=N_TRAIN
            )
            self.ct_sum.append(ct_sum_j)

        # Encrypt N for client to invert → gets 1/N
        N_padded = [float(N_TRAIN)] * self.slot_count
        pt_N     = encoder.encode(N_padded, SCALE)
        ct_N     = self.encryptor.encrypt(pt_N)

        print(f"  OK  Computed sum for {N_FEATURES} features")
        return self.ct_sum, ct_N

    # ─────────────────────────────────────────
    # STEP 2: Compute mu using 1/N from client
    # ct_mu_j = ct_sum_j * ct_inv_N
    # ─────────────────────────────────────────
    def compute_mu(self, ct_inv_N):
        """
        Computes mu for each feature.
        ct_mu_j = ct_sum_j * ct_inv_N

        Args:
            ct_inv_N : ciphertext of 1/N from client
        """
        print("[pca_algo.py] Computing mu...")

        self.ct_mu = []
        ct_inv_N_ref = Ciphertext(ct_inv_N)

        for j in range(N_FEATURES):
            ct_sum_j = Ciphertext(self.ct_sum[j])
            ct_inv   = Ciphertext(ct_inv_N_ref)

            # Align levels
            if ct_sum_j.coeff_modulus_size() > ct_inv.coeff_modulus_size():
                evaluator.mod_switch_to_inplace(ct_sum_j, ct_inv.parms_id())
            elif ct_inv.coeff_modulus_size() > ct_sum_j.coeff_modulus_size():
                evaluator.mod_switch_to_inplace(ct_inv, ct_sum_j.parms_id())
            ct_sum_j.scale(SCALE)
            ct_inv.scale(SCALE)

            ct_mu_j = evaluator.multiply(ct_sum_j, ct_inv)
            evaluator.relinearize_inplace(ct_mu_j, self.relin_keys)
            evaluator.rescale_to_next_inplace(ct_mu_j)
            ct_mu_j.scale(SCALE)
            self.ct_mu.append(ct_mu_j)

        print(f"  OK  mu computed for {N_FEATURES} features")

    # ─────────────────────────────────────────
    # STEP 3: Fit PCA components in plaintext
    # V is a global model parameter — acceptable to fit in plaintext
    # ─────────────────────────────────────────
    def fit_pca(self, train_csv_path, mu_plaintext):
        """
        Fits PCA on plaintext training data.
        V is a model parameter — not sensitive user data.

        Args:
            train_csv_path : path to normal_preprocessed.csv
            mu_plaintext   : numpy array of mu values (51,)
        """
        print(f"[pca_algo.py] Fitting PCA (n_components={self.n_components})...")

        df_train  = pd.read_csv(train_csv_path)
        X_train   = df_train.values.astype(np.float64)[:N_TRAIN]

        pca = PCA(n_components=self.n_components, svd_solver='full')
        pca.fit(X_train)

        self.V      = pca.components_   # (n_components, n_features)
        self.mu_pt  = mu_plaintext

        explained  = pca.explained_variance_ratio_
        print(f"  OK  PCA fitted. Explained variance: {explained.sum()*100:.1f}%")
        for k in range(self.n_components):
            print(f"      Component {k+1:2d}: {explained[k]*100:.2f}%")

        return self.V

    # ─────────────────────────────────────────
    # STEP 4: Compute PCA reconstruction error
    # For one encrypted test row
    # ─────────────────────────────────────────
    def compute_score(self, ct_x, pt_mu, pt_V_rows):
        """
        Computes PCA reconstruction error for one encrypted test row.

        score = || (I - V.T @ V) @ (x - mu) ||²

        Args:
            ct_x      : ciphertext with 51 feature values in slots 0..50
            pt_mu     : list of slot_count floats (mu padded)
            pt_V_rows : list of n_components arrays (each slot_count long)

        Returns:
            ciphertext where slot 0 = reconstruction error score
        """
        ct = Ciphertext(ct_x)
        ct.scale(SCALE)

        # ── Step 1: Center ct_xc = ct_x - mu ──────────────────
        _pt_mu = encoder.encode(pt_mu, SCALE)
        evaluator.mod_switch_to_inplace(_pt_mu, ct.parms_id())
        ct_xc = evaluator.sub_plain(ct, _pt_mu)

        # ── Step 2 & 3: Project + Reconstruct ─────────────────
        # ct_recon = sum_k [ dot(ct_xc, V[k]) * V[k] ]
        # dot(ct_xc, V[k]) = multiply_plain(ct_xc, V[k]) then slot_sum
        # result is scalar proj_k broadcast to all slots
        # ct_contrib_k = proj_k * V[k] = multiply_plain(ct_proj_k, V[k])

        ct_recon = None

        for k in range(self.n_components):
            # Encode V[k] at current scale
            _pt_vk = encoder.encode(pt_V_rows[k], ct_xc.scale())
            evaluator.mod_switch_to_inplace(_pt_vk, ct_xc.parms_id())

            # Project: ct_proj_k = ct_xc * V[k] elementwise then sum
            ct_proj = evaluator.multiply_plain(Ciphertext(ct_xc), _pt_vk)
            evaluator.rescale_to_next_inplace(ct_proj)
            ct_proj.scale(SCALE)

            # slot_sum → proj_k broadcast to all slots
            ct_proj = self.slot_sum(ct_proj, ROTATION_STEPS_FEAT)

            # Reconstruct contribution: proj_k * V[k]
            _pt_vk2 = encoder.encode(pt_V_rows[k], ct_proj.scale())
            evaluator.mod_switch_to_inplace(_pt_vk2, ct_proj.parms_id())
            ct_contrib = evaluator.multiply_plain(ct_proj, _pt_vk2)
            evaluator.rescale_to_next_inplace(ct_contrib)
            ct_contrib.scale(SCALE)

            # Accumulate reconstruction
            if ct_recon is None:
                ct_recon = ct_contrib
            else:
                evaluator.mod_switch_to_inplace(ct_recon, ct_contrib.parms_id())
                ct_recon.scale(ct_contrib.scale())
                evaluator.add_inplace(ct_recon, ct_contrib)

        # ── Step 4: Residual = ct_xc - ct_recon ───────────────
        evaluator.mod_switch_to_inplace(ct_xc, ct_recon.parms_id())
        ct_xc.scale(ct_recon.scale())
        ct_resid = evaluator.sub(ct_xc, ct_recon)

        # ── Step 5: Square the residual ───────────────────────
        ct_err2 = evaluator.square(ct_resid)
        evaluator.relinearize_inplace(ct_err2, self.relin_keys)
        evaluator.rescale_to_next_inplace(ct_err2)
        ct_err2.scale(SCALE)

        # ── Step 6: Sum 51 error slots → scalar ───────────────
        ct_score = self.slot_sum(ct_err2, ROTATION_STEPS_FEAT)

        return ct_score

    # ─────────────────────────────────────────
    # Batch scoring for all test rows
    # ─────────────────────────────────────────
    def compute_scores_batch(self, test_ciphertexts, mu_plaintext, V):
        """
        Computes PCA reconstruction error for all test rows.

        Args:
            test_ciphertexts : list of N_TEST seal.Ciphertext (row-major)
            mu_plaintext     : numpy array (51,)
            V                : numpy array (n_components, 51)

        Returns:
            list of N_TEST ciphertexts (each = encrypted score)
        """
        print(f"[pca_algo.py] Computing PCA scores for {len(test_ciphertexts)} test rows...")

        slot_count = encoder.slot_count()

        # Prepare padded mu and V rows
        pt_mu = mu_plaintext.tolist() + [0.0] * (slot_count - len(mu_plaintext))

        pt_V_rows = []
        for k in range(self.n_components):
            v_row = V[k].tolist() + [0.0] * (slot_count - len(V[k]))
            pt_V_rows.append(v_row)

        score_cts = []

        for i, ct_x in enumerate(test_ciphertexts):
            ct_score = self.compute_score(ct_x, pt_mu, pt_V_rows)
            score_cts.append(ct_score)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(test_ciphertexts)}] scores computed")

        print(f"  OK  {len(score_cts)} PCA scores computed")
        return score_cts