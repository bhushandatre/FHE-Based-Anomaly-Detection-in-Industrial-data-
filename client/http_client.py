"""
client/http_client.py
=====================
HTTP client that runs the full HE protocol over the network.
Talks to server/http_server.py via REST API.

All HE computation stays on the server.
Client only handles:
  - encrypting data
  - inversion rounds (decrypt → 1/x → re-encrypt)
  - final score decryption and decision
"""

import sys
import os
import time
import requests
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from key import context, encoder, SCALE, load_client_keys

from client.decrypt import decrypt_scalar, decrypt_vector
from client.invert  import invert_scalar, invert_vector
from client.decide  import decide
from shared.config  import (
    SERVER_URL, REQUEST_TIMEOUT, N_FEATURES,
    N_TRAIN, N_TEST, THRESHOLD_N,
    TRAIN_CT_DIR, TEST_CT_DIR, LABELS_PATH,
    TRAIN_CSV
)
from shared.protocol import (
    cts_to_payload, payload_to_cts,
    single_ct_payload, payload_to_single_ct
)
from seal import Ciphertext


class HEClient:
    """
    Runs the full HE protocol over HTTP.
    Instantiate, call run_sad() or run_pca().
    """

    def __init__(self):
        self.round_times = {}
        self.scores      = None
        self.predictions = None

    # ─────────────────────────────────────────
    # HTTP HELPERS
    # ─────────────────────────────────────────
    def _post(self, endpoint: str, data: dict) -> dict:
        url = SERVER_URL + endpoint
        try:
            r = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  ERROR POST {endpoint}: {e}")
            raise

    def _get(self, endpoint: str) -> dict:
        url = SERVER_URL + endpoint
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  ERROR GET {endpoint}: {e}")
            raise

    def _time_round(self, name: str, fn):
        t0 = time.time()
        result = fn()
        elapsed = (time.time() - t0) * 1000
        self.round_times[name] = round(elapsed, 2)
        print(f"  OK  {name} — {elapsed:.0f}ms")
        return result

    # ─────────────────────────────────────────
    # CHECK SERVER
    # ─────────────────────────────────────────
    def check_server(self) -> bool:
        try:
            r = requests.get(SERVER_URL + "/api/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ─────────────────────────────────────────
    # LOAD ENCRYPTED CIPHERTEXTS FROM DISK
    # ─────────────────────────────────────────
    def _load_train_cts(self):
        cts = []
        for j in range(N_FEATURES):
            ct = Ciphertext()
            ct.load(context, os.path.join(TRAIN_CT_DIR, f"ct_feature_{j:02d}.seal"))
            cts.append(ct)
        return cts

    def _load_test_cts(self):
        cts = []
        for i in range(N_TEST):
            ct = Ciphertext()
            ct.load(context, os.path.join(TEST_CT_DIR, f"ct_{i:06d}.seal"))
            cts.append(ct)
        return cts

    # ─────────────────────────────────────────
    # THRESHOLD COMPUTATION
    # ─────────────────────────────────────────
    def _compute_threshold_sad(self, mu_padded, inv_sigma2_padded):
        from server.algo import SAD
        sad_local    = SAD()
        encryptor_t, decryptor_t, _, _ = load_client_keys()
        df_train     = pd.read_csv(TRAIN_CSV)
        X_train      = df_train.values.astype(np.float64)[:N_TRAIN]
        slot_count   = encoder.slot_count()
        train_scores = []

        for i in range(THRESHOLD_N):
            row    = X_train[i].tolist()
            padded = row + [0.0] * (slot_count - len(row))
            plain  = encoder.encode(padded, SCALE)
            ct_i   = encryptor_t.encrypt(plain)
            sc     = sad_local.compute_scores_batch([ct_i], mu_padded, inv_sigma2_padded)
            dec    = encoder.decode(decryptor_t.decrypt(sc[0]))
            train_scores.append(float(np.real(dec[0])))

        train_scores = np.array(train_scores)
        threshold    = float(np.mean(train_scores) + 3 * np.std(train_scores))
        print(f"  OK  Threshold: {threshold:.6f}")
        return threshold

    def _compute_threshold_pca(self, mu_plaintext, V):
        from server.pca_algo import PCA_HE
        pca_local    = PCA_HE(n_components=V.shape[0])
        encryptor_t, decryptor_t, _, _ = load_client_keys()
        df_train     = pd.read_csv(TRAIN_CSV)
        X_train      = df_train.values.astype(np.float64)[:N_TRAIN]
        slot_count   = encoder.slot_count()

        pt_mu  = mu_plaintext.tolist() + [0.0] * (slot_count - len(mu_plaintext))
        pt_V   = [V[k].tolist() + [0.0]*(slot_count - V.shape[1]) for k in range(V.shape[0])]

        train_scores = []
        for i in range(THRESHOLD_N):
            row    = X_train[i].tolist()
            padded = row + [0.0] * (slot_count - len(row))
            plain  = encoder.encode(padded, SCALE)
            ct_i   = encryptor_t.encrypt(plain)
            sc     = pca_local.compute_score(ct_i, pt_mu, pt_V)
            dec    = encoder.decode(decryptor_t.decrypt(sc))
            train_scores.append(float(np.real(dec[0])))

        train_scores = np.array(train_scores)
        threshold    = float(np.mean(train_scores) + 3 * np.std(train_scores))
        print(f"  OK  Threshold: {threshold:.6f}")
        return threshold

    # ─────────────────────────────────────────
    # RUN SAD PROTOCOL
    # ─────────────────────────────────────────
    def run_sad(self):
        print("\n" + "="*60)
        print("HE-SAD Protocol — Network Mode")
        print("="*60)

        if not self.check_server():
            print("  ERROR: Server not reachable at", SERVER_URL)
            return

        # Reset server
        self._post("/api/reset", {"mode": "SAD"})

        # Round 1 — send training data
        print("\n[Round 1] Sending training data...")
        def send_train():
            cts     = self._load_train_cts()
            payload = cts_to_payload(cts, label="train")
            return self._post("/api/upload/train", payload)
        self._time_round("Round 1 — Send training data", send_train)

        # Round 2 — get ct_N from server
        print("\n[Round 2] Server computing sums...")
        def get_sum():
            return self._get("/api/compute/sum")
        resp = self._time_round("Round 2 — Compute feature sums", get_sum)
        ct_N = payload_to_single_ct(resp, context)

        # Round 3 — invert N, send back
        print("\n[Round 3] Inverting N...")
        def invert_N():
            value    = decrypt_scalar(ct_N)
            print(f"  OK  Decrypted N = {value:.0f}")
            ct_inv_N = invert_scalar(value)
            payload  = single_ct_payload(ct_inv_N, label="ct_inv_N")
            return self._post("/api/invert/N", payload)
        self._time_round("Round 3 — Invert N", invert_N)

        # Round 4 — server computes mu + sigma², get ct_sigma²
        print("\n[Round 4] Server computing mu and sigma²...")
        def get_sigma2():
            return self._get("/api/compute/mu")
        resp    = self._time_round("Round 4 — Compute mu and sigma²", get_sigma2)
        ct_sigma2 = payload_to_single_ct(resp, context)

        # Round 5 — invert sigma², send back
        print("\n[Round 5] Inverting sigma²...")
        def invert_sigma2():
            values      = decrypt_vector(ct_sigma2, N_FEATURES)
            print(f"  OK  sigma² range: [{values.min():.4f}, {values.max():.4f}]")
            ct_inv_s2   = invert_vector(values)
            payload     = single_ct_payload(ct_inv_s2, label="ct_inv_sigma2")
            return self._post("/api/invert/sigma2", payload)
        self._time_round("Round 5 — Invert sigma²", invert_sigma2)

        # Round 6 — send test data in batches
        print("\n[Round 6] Sending test data in batches...")
        def send_test():
            cts       = self._load_test_cts()
            batch_size = 100
            total      = len(cts)
            for start in range(0, total, batch_size):
                batch   = cts[start:start+batch_size]
                payload = cts_to_payload(batch, label=f"test_batch_{start}")
                payload["batch_start"] = start
                payload["batch_total"] = total
                self._post("/api/upload/test", payload)
                print(f"  [{min(start+batch_size, total)}/{total}] batches sent")
            return {"status": "ok", "received": total}
        self._time_round("Round 6 — Send test data", send_test)

        # Round 7 — trigger score computation (runs in background on server)
        print("\n[Round 7] Server computing scores...")
        def get_scores():
            return self._get("/api/compute/scores")
        self._time_round("Round 7 — Start score computation", get_scores)

        # Poll until scores are ready
        print("  Waiting for server to finish computing scores...")
        from shared.config import SCORE_TIMEOUT
        import time as _time
        t0 = _time.time()
        while True:
            poll = self._get("/api/scores/status")
            if poll["status"] == "ready":
                total = poll["count"]
                print(f"  OK  {total} scores ready")
                break
            elapsed = _time.time() - t0
            if elapsed > SCORE_TIMEOUT:
                raise TimeoutError(f"Score computation timed out after {SCORE_TIMEOUT}s")
            print(f"  Computing... ({int(elapsed)}s elapsed)")
            _time.sleep(5)

        # Round 8 — fetch scores in batches
        print("\n[Round 8] Fetching scores in batches...")
        score_cts  = []
        batch_size = 100
        def fetch_batch():
            for start in range(0, total, batch_size):
                r    = self._get(f"/api/scores/batch?start={start}&size={batch_size}")
                batch = payload_to_cts(r, context)
                score_cts.extend(batch)
                print(f"  [{min(start+batch_size, total)}/{total}] scores fetched")
            return score_cts
        self._time_round("Round 8 — Fetch scores", fetch_batch)

        # Round 9 — decrypt and decide
        print("\n[Round 9] Decrypting scores and deciding...")
        df_train     = pd.read_csv(TRAIN_CSV)
        X_train      = df_train.values.astype(np.float64)[:N_TRAIN]
        mu_vals      = X_train.mean(axis=0).tolist()
        sigma2_vals  = np.var(X_train, axis=0)
        inv_sigma2   = [1.0 / max(float(v), 0.01) for v in sigma2_vals]
        slot_count   = encoder.slot_count()
        mu_padded    = mu_vals + [0.0] * (slot_count - len(mu_vals))
        inv_s2_padded = inv_sigma2 + [0.0] * (slot_count - len(inv_sigma2))

        threshold = self._compute_threshold_sad(mu_padded, inv_s2_padded)

        predictions, scores = decide(
            score_cts, threshold, labels_path=LABELS_PATH
        )
        self.scores      = scores
        self.predictions = predictions

        # Post final results back to server for dashboard
        self._post_results(scores, predictions)

        print("\n" + "="*60)
        print("HE-SAD COMPLETE")
        print("="*60)
        self._print_summary()
        return scores, predictions

    # ─────────────────────────────────────────
    # RUN PCA PROTOCOL
    # ─────────────────────────────────────────
    def run_pca(self):
        print("\n" + "="*60)
        print("HE-PCA Protocol — Network Mode")
        print("="*60)

        if not self.check_server():
            print("  ERROR: Server not reachable at", SERVER_URL)
            return

        self._post("/api/reset", {"mode": "PCA"})

        # Round 1 — training data
        print("\n[Round 1] Sending training data...")
        def send_train():
            cts     = self._load_train_cts()
            payload = cts_to_payload(cts, label="train")
            return self._post("/api/upload/train", payload)
        self._time_round("Round 1 — Send training data", send_train)

        # Round 2 — get ct_N
        print("\n[Round 2] Server computing sums...")
        resp = self._time_round("Round 2 — Compute feature sums",
                                lambda: self._get("/api/compute/sum"))
        ct_N = payload_to_single_ct(resp, context)

        # Round 3 — invert N
        print("\n[Round 3] Inverting N...")
        def invert_N():
            value    = decrypt_scalar(ct_N)
            print(f"  OK  Decrypted N = {value:.0f}")
            ct_inv_N = invert_scalar(value)
            payload  = single_ct_payload(ct_inv_N, label="ct_inv_N")
            return self._post("/api/invert/N", payload)
        self._time_round("Round 3 — Invert N", invert_N)

        # Round 4 — server fits PCA with optimal k
        print("\n[Round 4] Server fitting PCA components...")
        resp = self._time_round("Round 4 — Fit PCA",
                                lambda: self._get("/api/compute/mu"))
        n_components = resp.get("n_components", 10)
        var_explained = resp.get("variance_explained", 0)
        print(f"  OK  Optimal k={n_components} ({var_explained*100:.2f}% variance explained)")

        # Round 5 — send test data in batches
        print("\n[Round 5] Sending test data in batches...")
        def send_test():
            cts        = self._load_test_cts()
            batch_size = 100
            total      = len(cts)
            for start in range(0, total, batch_size):
                batch   = cts[start:start+batch_size]
                payload = cts_to_payload(batch, label=f"test_batch_{start}")
                payload["batch_start"] = start
                payload["batch_total"] = total
                self._post("/api/upload/test", payload)
                print(f"  [{min(start+batch_size, total)}/{total}] batches sent")
            return {"status": "ok", "received": total}
        self._time_round("Round 5 — Send test data", send_test)

        # Round 6 — trigger score computation (runs in background on server)
        print("\n[Round 6] Server computing PCA scores...")
        self._time_round("Round 6 — Start score computation",
                         lambda: self._get("/api/compute/scores"))

        # Poll until scores are ready
        print("  Waiting for server to finish computing PCA scores...")
        from shared.config import SCORE_TIMEOUT
        import time as _time
        t0 = _time.time()
        while True:
            poll = self._get("/api/scores/status")
            if poll["status"] == "ready":
                total = poll["count"]
                print(f"  OK  {total} scores ready")
                break
            elapsed = _time.time() - t0
            if elapsed > SCORE_TIMEOUT:
                raise TimeoutError(f"Score computation timed out after {SCORE_TIMEOUT}s")
            print(f"  Computing... ({int(elapsed)}s elapsed)")
            _time.sleep(5)

        # Round 7 — fetch scores in batches
        print("\n[Round 7] Fetching scores in batches...")
        score_cts  = []
        batch_size = 100
        def fetch_batch():
            for start in range(0, total, batch_size):
                r     = self._get(f"/api/scores/batch?start={start}&size={batch_size}")
                batch = payload_to_cts(r, context)
                score_cts.extend(batch)
                print(f"  [{min(start+batch_size, total)}/{total}] scores fetched")
            return score_cts
        self._time_round("Round 7 — Fetch scores", fetch_batch)

        # Round 8 — decrypt and decide
        print("\n[Round 8] Decrypting scores and deciding...")
        df_train      = pd.read_csv(TRAIN_CSV)
        X_train       = df_train.values.astype(np.float64)[:N_TRAIN]
        mu_plaintext  = X_train.mean(axis=0)

        # Use same optimal k as server used
        from sklearn.decomposition import PCA as SklearnPCA
        from shared.config import VARIANCE_THRESH
        pca_full   = SklearnPCA()
        pca_full.fit(X_train)
        cumvar     = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_k  = int(np.argmax(cumvar >= VARIANCE_THRESH) + 1)
        print(f"  OK  Using k={optimal_k} components for threshold computation")
        pca = SklearnPCA(n_components=optimal_k)
        pca.fit(X_train)
        V = pca.components_

        threshold = self._compute_threshold_pca(mu_plaintext, V)

        predictions, scores = decide(
            score_cts, threshold, labels_path=LABELS_PATH
        )
        self.scores      = scores
        self.predictions = predictions

        # Post final results back to server for dashboard
        self._post_results(scores, predictions)

        print("\n" + "="*60)
        print("HE-PCA COMPLETE")
        print("="*60)
        self._print_summary()
        return scores, predictions

    # ─────────────────────────────────────────
    # POST RESULTS TO SERVER FOR DASHBOARD
    # ─────────────────────────────────────────
    def _post_results(self, scores, predictions):
        import numpy as np
        from sklearn.metrics import (
            f1_score, roc_auc_score, precision_score,
            recall_score, confusion_matrix
        )
        y_true = np.load(LABELS_PATH)
        auc    = float(roc_auc_score(y_true, scores))
        f1     = float(f1_score(y_true, predictions))
        prec   = float(precision_score(y_true, predictions, zero_division=0))
        rec    = float(recall_score(y_true, predictions, zero_division=0))
        cm     = confusion_matrix(y_true, predictions)

        payload = {
            "metrics": {
                "roc_auc"  : round(auc,  4),
                "f1"       : round(f1,   4),
                "precision": round(prec, 4),
                "recall"   : round(rec,  4),
            },
            "confusion_matrix": {
                "TN": int(cm[0,0]), "FP": int(cm[0,1]),
                "FN": int(cm[1,0]), "TP": int(cm[1,1])
            },
            "scores": scores.tolist(),
            "labels": y_true.tolist()
        }
        try:
            self._post("/api/results", payload)
            print("  OK  Results posted to server for dashboard")
        except Exception as e:
            print(f"  WARNING: Could not post results: {e}")

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    def _print_summary(self):
        print(f"\n  {'Round':<40} {'Time (ms)':>10}")
        print(f"  {'-'*52}")
        for name, ms in self.round_times.items():
            print(f"  {name:<40} {ms:>10.0f}")
        total = sum(self.round_times.values())
        print(f"  {'-'*52}")
        print(f"  {'TOTAL':<40} {total:>10.0f}")