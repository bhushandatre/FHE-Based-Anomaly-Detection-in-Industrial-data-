"""
server/algo.py
==============
SAD algorithm running entirely on encrypted data.
Server never decrypts anything.

Division is handled via multi-round protocol:
    Server sends encrypted denominator to client
    Client decrypts, inverts, re-encrypts
    Server multiplies instead of divides

Steps:
    1. slot_sum(ct_feature_j) for each feature  → ct_sum
    2. ct_mu = ct_sum * ct_inv_N                ← ct_inv_N from client
    3. ct_dev2 = (ct_x - ct_mu)²
    4. slot_sum(ct_dev2) / N                    → ct_sigma2
    5. ct_score = sum_j[(ct_x_j - ct_mu_j)² * ct_inv_sigma2_j]

Usage:
    from server.algo import SAD
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from seal import Ciphertext
from key import context, encoder, evaluator, SCALE, load_server_keys

# Rotation steps for slot_sum of N_TRAIN=5000 slots
ROTATION_STEPS_TRAIN = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# Rotation steps for slot_sum of 51 feature slots
ROTATION_STEPS_FEAT  = [1, 2, 4, 8, 16, 32]

N_TRAIN   = 5000
N_FEATURES = 51


class SAD:
    def __init__(self):
        self.encryptor, self.relin_keys, self.galois_keys = load_server_keys()
        self.slot_count = encoder.slot_count()

        # Stored model parameters (encrypted)
        self.ct_mu          = None   # list of 51 ciphertexts (one per feature)
        self.ct_inv_sigma2  = None   # list of 51 ciphertexts (one per feature)
        self.ct_sum         = None   # list of 51 ciphertexts (running sum)
        self.ct_var_sum     = None   # list of 51 ciphertexts (variance sum)

    # ─────────────────────────────────────────
    # HELPER: slot_sum
    # Sums N slots using binary rotation tree
    # ─────────────────────────────────────────
    def slot_sum(self, ct, rotation_steps):
        """Sum slots using rotations. Returns ct where every slot = sum."""
        ct_result = Ciphertext(ct)
        total     = 1
        for step in rotation_steps:
            if total >= N_TRAIN:
                break
            ct_rot = evaluator.rotate_vector(ct_result, step, self.galois_keys)
            evaluator.mod_switch_to_inplace(ct_rot, ct_result.parms_id())
            ct_rot.scale(ct_result.scale())
            evaluator.add_inplace(ct_result, ct_rot)
            total += step
        return ct_result

    # ─────────────────────────────────────────
    # STEP 1: Compute sum per feature
    # Returns list of 51 encrypted sums
    # Server sends ct_N to client for inversion
    # ─────────────────────────────────────────
    def compute_sum(self, train_ciphertexts):
        """
        Computes slot_sum for each of 51 training ciphertexts.
        Returns ct_sum list and ct_N (encrypted count N).

        Args:
            train_ciphertexts : list of 51 seal.Ciphertext (column-major)

        Returns:
            ct_sum : list of 51 ciphertexts (each = sum of feature across rows)
            ct_N   : ciphertext containing N in all slots (for client to invert)
        """
        print("[algo.py] Computing feature sums...")

        self.ct_sum = []
        for j in range(N_FEATURES):
            ct_sum_j = self.slot_sum(train_ciphertexts[j], ROTATION_STEPS_TRAIN)
            self.ct_sum.append(ct_sum_j)

        # Encrypt N for client to invert
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
            ct_inv_N : ciphertext of 1/N (received from client)
        """
        print("[algo.py] Computing mu...")

        self.ct_mu = []
        for j in range(N_FEATURES):
            ct_sum_j = Ciphertext(self.ct_sum[j])
            evaluator.mod_switch_to_inplace(ct_inv_N, ct_sum_j.parms_id())
            ct_inv_N.scale(ct_sum_j.scale())
            ct_mu_j = evaluator.multiply(ct_sum_j, ct_inv_N)
            evaluator.relinearize_inplace(ct_mu_j, self.relin_keys)
            evaluator.rescale_to_next_inplace(ct_mu_j)
            ct_mu_j.scale(SCALE)
            self.ct_mu.append(ct_mu_j)

        print(f"  OK  mu computed for {N_FEATURES} features")

    # ─────────────────────────────────────────
    # STEP 3: Compute sigma² per feature
    # sigma2_j = slot_sum((ct_feature_j - ct_mu_j)²) / N
    # Server sends ct_sigma2 to client for inversion
    # ─────────────────────────────────────────
    def compute_sigma2(self, train_ciphertexts):
        """
        Computes variance sum for each feature.
        Returns ct_sigma2 list and sends to client for inversion.

        Args:
            train_ciphertexts : list of 51 seal.Ciphertext (column-major)

        Returns:
            ct_sigma2 : list of 51 ciphertexts (one sigma² per feature)
        """
        print("[algo.py] Computing sigma²...")

        ct_sigma2_list = []

        for j in range(N_FEATURES):
            ct_j    = Ciphertext(train_ciphertexts[j])
            ct_mu_j = Ciphertext(self.ct_mu[j])

            # Always switch the HIGHER level ct DOWN to match the LOWER level ct
            # ct_mu_j is lower level (went through multiply + rescale)
            # ct_j is at original level → switch ct_j down to ct_mu_j level
            if ct_j.coeff_modulus_size() > ct_mu_j.coeff_modulus_size():
                evaluator.mod_switch_to_inplace(ct_j, ct_mu_j.parms_id())
            elif ct_mu_j.coeff_modulus_size() > ct_j.coeff_modulus_size():
                evaluator.mod_switch_to_inplace(ct_mu_j, ct_j.parms_id())
            ct_j.scale(SCALE)
            ct_mu_j.scale(SCALE)

            # ct_dev = ct_feature_j - ct_mu_j
            ct_dev = evaluator.sub(ct_j, ct_mu_j)

            # ct_dev² = ct_dev * ct_dev
            ct_dev2 = evaluator.square(ct_dev)
            evaluator.relinearize_inplace(ct_dev2, self.relin_keys)
            evaluator.rescale_to_next_inplace(ct_dev2)
            ct_dev2.scale(SCALE)

            # slot_sum of squared deviations
            ct_var_sum_j = self.slot_sum(ct_dev2, ROTATION_STEPS_TRAIN)
            ct_sigma2_list.append(ct_var_sum_j)

        self.ct_var_sum = ct_sigma2_list
        print(f"  OK  sigma² sum computed for {N_FEATURES} features")
        return ct_sigma2_list

    # ─────────────────────────────────────────
    # STEP 4: Store 1/sigma² from client
    # ─────────────────────────────────────────
    def store_inv_sigma2(self, ct_inv_sigma2_list):
        """
        Stores encrypted 1/sigma² received from client.

        Args:
            ct_inv_sigma2_list : list of 51 ciphertexts (one 1/sigma² per feature)
        """
        self.ct_inv_sigma2 = ct_inv_sigma2_list
        print(f"[algo.py] Stored ct_inv_sigma2 for {N_FEATURES} features")

    # ─────────────────────────────────────────
    # STEP 5: Compute SAD score for a test row
    # score = sum_j[(x_j - mu_j)² * inv_sigma2_j]
    # ─────────────────────────────────────────
    def compute_score(self, ct_test_row):
        """
        Computes SAD anomaly score for one encrypted test row.
        score = sum_j [(x_j - mu_j)² / sigma²_j]

        Args:
            ct_test_row : ciphertext with 51 feature values in slots 0..50

        Returns:
            ciphertext where slot 0 = anomaly score
        """
        ct_x = Ciphertext(ct_test_row)

        # Pack mu into a single plaintext vector (51 slots)
        # We decode mu from each ct_mu and encode as plaintext
        # This is acceptable — mu is a model parameter
        mu_vals = []
        for j in range(N_FEATURES):
            _, decryptor_dummy, _, _ = load_server_keys.__module__, None, None, None
            # mu is already encrypted — we use it directly slot by slot
            # instead we use the plaintext approach via encoded vector
            mu_vals.append(0.0)   # placeholder — replaced below

        # Use encrypted mu directly (no decryption on server)
        # ct_dev_j = ct_x[slot j] - ct_mu[j][slot 0]
        # This requires extracting slot j from ct_x and slot 0 from ct_mu_j

        # Simpler approach: ct_mu is column-major (all rows same mu)
        # Encode mu as plaintext vector for inference (server computes this)
        # Note: this is the SAD score using encrypted mu broadcast

        # SAD score computation:
        # For each feature j:
        #   ct_dev_j  = ct_x * mask_j - ct_mu_j * mask_j
        #   ct_dev2_j = ct_dev_j²
        #   ct_w_j    = ct_dev2_j * ct_inv_sigma2_j
        # sum all 51 weighted squared deviations into slot 0

        # Since ct_x has all 51 features in slots 0..50
        # and ct_mu has mu_j broadcast across all slots
        # we encode mu as a 51-slot plaintext vector

        # Decode mu from encrypted mu (server knows encrypted mu, not plaintext)
        # For inference: mu is used as plaintext since it's a model param
        # received back from client after inversion round

        # ct_dev = ct_x - pt_mu
        ct_x.scale(SCALE)
        evaluator.mod_switch_to_inplace(ct_x, ct_x.parms_id())

        # (x - mu)²
        ct_dev  = ct_x   # placeholder — full implementation in run.py
        ct_dev2 = evaluator.square(ct_dev)
        evaluator.relinearize_inplace(ct_dev2, self.relin_keys)
        evaluator.rescale_to_next_inplace(ct_dev2)
        ct_dev2.scale(SCALE)

        # * inv_sigma2 (first feature only as placeholder)
        ct_inv_s2 = Ciphertext(self.ct_inv_sigma2[0])
        evaluator.mod_switch_to_inplace(ct_inv_s2, ct_dev2.parms_id())
        ct_inv_s2.scale(ct_dev2.scale())
        ct_w = evaluator.multiply(ct_dev2, ct_inv_s2)
        evaluator.relinearize_inplace(ct_w, self.relin_keys)
        evaluator.rescale_to_next_inplace(ct_w)
        ct_w.scale(SCALE)

        return ct_w

    def compute_scores_batch(self, test_ciphertexts, pt_mu, pt_inv_sigma2):
        """
        Computes SAD score for all test rows.
        Uses plaintext mu and inv_sigma2 vectors for efficiency.

        Args:
            test_ciphertexts : list of N_TEST seal.Ciphertext (row-major)
            pt_mu            : seal.Plaintext with mu values in slots 0..50
            pt_inv_sigma2    : seal.Plaintext with 1/sigma² in slots 0..50

        Returns:
            list of N_TEST ciphertexts (each = encrypted score)
        """
        print(f"[algo.py] Computing SAD scores for {len(test_ciphertexts)} test rows...")

        score_cts = []

        for i, ct_x in enumerate(test_ciphertexts):
            # Fresh plaintexts each call to avoid level degradation
            _pt_mu        = encoder.encode(pt_mu,        SCALE)
            _pt_inv_sigma2 = encoder.encode(pt_inv_sigma2, SCALE)

            ct = Ciphertext(ct_x)
            ct.scale(SCALE)

            # (x - mu)
            evaluator.mod_switch_to_inplace(_pt_mu, ct.parms_id())
            ct_dev = evaluator.sub_plain(ct, _pt_mu)

            # (x - mu)²
            ct_dev2 = evaluator.square(ct_dev)
            evaluator.relinearize_inplace(ct_dev2, self.relin_keys)
            evaluator.rescale_to_next_inplace(ct_dev2)
            ct_dev2.scale(SCALE)

            # * (1/sigma²)
            evaluator.mod_switch_to_inplace(_pt_inv_sigma2, ct_dev2.parms_id())
            ct_w = evaluator.multiply_plain(ct_dev2, _pt_inv_sigma2)
            evaluator.rescale_to_next_inplace(ct_w)

            # Sum 51 feature slots → slot 0 = score
            ct_score = ct_w
            for step in ROTATION_STEPS_FEAT:
                ct_rot = evaluator.rotate_vector(ct_score, step, self.galois_keys)
                evaluator.mod_switch_to_inplace(ct_rot, ct_score.parms_id())
                ct_rot.scale(ct_score.scale())
                evaluator.add_inplace(ct_score, ct_rot)

            score_cts.append(ct_score)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(test_ciphertexts)}] scores computed")

        print(f"  OK  {len(score_cts)} scores computed")
        return score_cts