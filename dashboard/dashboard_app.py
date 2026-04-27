"""
dashboard/dashboard_app.py
===========================
Flask + SocketIO dashboard that controls the HE client directly.

Run only TWO terminals:
  Terminal 1: uvicorn server.http_server:app --host 0.0.0.0 --port 8000
  Terminal 2: python dashboard/dashboard_app.py

Then open http://localhost:3000 and use the buttons.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import threading
import time
import numpy as np

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from shared.config import SERVER_URL, DASHBOARD_PORT, SAD_ROUNDS, PCA_ROUNDS

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = 'he_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────
# CLIENT STATE
# ─────────────────────────────────────────────
client_state = {
    "phase"    : "idle",
    "progress" : 0,
    "message"  : "Ready — click Send Data to begin",
    "paused"   : False,
    "stop"     : False,
    "algorithm": None,
    "error"    : None,
}
state_lock = threading.Lock()

def update_state(**kwargs):
    with state_lock:
        client_state.update(kwargs)
    socketio.emit("client_state", dict(client_state))

def log(msg):
    socketio.emit("log", {"msg": msg})
    print(f"[LOG] {msg}")

# ─────────────────────────────────────────────
# SERVER POLLER
# ─────────────────────────────────────────────
def poll_server():
    while True:
        try:
            r = requests.get(SERVER_URL + "/api/status", timeout=3)
            if r.status_code == 200:
                socketio.emit("status_update", r.json())
                if r.json().get("status") in ("scores_ready", "complete"):
                    rr = requests.get(SERVER_URL + "/api/results", timeout=3)
                    if rr.status_code == 200:
                        socketio.emit("results_update", rr.json())
        except Exception:
            socketio.emit("server_offline", {"message": "HE server offline"})
        time.sleep(1)

threading.Thread(target=poll_server, daemon=True).start()

# ─────────────────────────────────────────────
# UPLOAD THREAD
# ─────────────────────────────────────────────
def _do_upload():
    from seal import Ciphertext
    from key import context
    from shared.config import N_FEATURES, N_TEST, TRAIN_CT_DIR, TEST_CT_DIR
    from shared.protocol import cts_to_payload

    try:
        update_state(phase="uploading", progress=0,
                     message="Resetting server...")
        requests.post(SERVER_URL + "/api/reset",
                      json={"mode": "SAD"}, timeout=10)

        # Training ciphertexts
        log("Loading training ciphertexts...")
        train_cts = []
        for j in range(N_FEATURES):
            if client_state["stop"]:
                update_state(phase="idle", message="Stopped", progress=0)
                return
            while client_state["paused"]:
                time.sleep(0.3)
            ct = Ciphertext()
            ct.load(context, os.path.join(TRAIN_CT_DIR,
                                          f"ct_feature_{j:02d}.seal"))
            train_cts.append(ct)

        log("Uploading training data to server...")
        r = requests.post(SERVER_URL + "/api/upload/train",
                          json=cts_to_payload(train_cts, "train"),
                          timeout=120)
        r.raise_for_status()
        update_state(progress=10,
                     message="Training data uploaded ✓")
        log("Training data uploaded ✓")

        # Test ciphertexts in batches
        batch_size = 100
        for start in range(0, N_TEST, batch_size):
            if client_state["stop"]:
                update_state(phase="idle", message="Stopped", progress=0)
                return
            while client_state["paused"]:
                update_state(message=f"Paused at {start}/{N_TEST} test rows")
                time.sleep(0.3)

            batch = []
            for i in range(start, min(start + batch_size, N_TEST)):
                ct = Ciphertext()
                ct.load(context,
                        os.path.join(TEST_CT_DIR, f"ct_{i:06d}.seal"))
                batch.append(ct)

            p = cts_to_payload(batch, f"test_{start}")
            p["batch_start"] = start
            p["batch_total"] = N_TEST
            requests.post(SERVER_URL + "/api/upload/test",
                          json=p, timeout=120).raise_for_status()

            done = min(start + batch_size, N_TEST)
            pct  = 10 + int(done / N_TEST * 90)
            update_state(progress=pct,
                         message=f"Test data: {done}/{N_TEST} sent")

        update_state(phase="uploaded", progress=100,
                     message="All data uploaded ✓  Select an algorithm below")
        log("Upload complete ✓")

    except Exception as e:
        update_state(phase="error", error=str(e),
                     message=f"Upload failed: {e}")
        log(f"ERROR during upload: {e}")

# ─────────────────────────────────────────────
# ALGORITHM THREAD
# ─────────────────────────────────────────────
def _do_run_algorithm(algorithm: str):
    from seal import Ciphertext
    from key import context, encoder, SCALE, load_client_keys
    from client.decrypt import decrypt_scalar, decrypt_vector
    from client.invert  import invert_scalar, invert_vector
    from client.decide  import decide
    from shared.config  import (
        N_TRAIN, N_TEST, N_FEATURES, THRESHOLD_N,
        TRAIN_CSV, LABELS_PATH, SCORE_TIMEOUT,
        TRAIN_CT_DIR, TEST_CT_DIR, VARIANCE_THRESH
    )
    from shared.protocol import (
        cts_to_payload, single_ct_payload,
        payload_to_single_ct, payload_to_cts
    )
    import pandas as pd

    try:
        update_state(phase="running", algorithm=algorithm,
                     message=f"Starting {algorithm} protocol...")

        # Reset server with correct algorithm mode
        log(f"Resetting server for {algorithm} mode...")
        requests.post(SERVER_URL + "/api/reset",
                      json={"mode": algorithm}, timeout=10)

        # Re-upload training data
        log("Re-uploading training data...")
        train_cts = []
        for j in range(N_FEATURES):
            ct = Ciphertext()
            ct.load(context, os.path.join(TRAIN_CT_DIR,
                                          f"ct_feature_{j:02d}.seal"))
            train_cts.append(ct)
        requests.post(SERVER_URL + "/api/upload/train",
                      json=cts_to_payload(train_cts, "train"),
                      timeout=120).raise_for_status()
        log("Training data re-uploaded ✓")

        # Re-upload test data
        log("Re-uploading test data...")
        for start in range(0, N_TEST, 100):
            batch = []
            for i in range(start, min(start + 100, N_TEST)):
                ct = Ciphertext()
                ct.load(context,
                        os.path.join(TEST_CT_DIR, f"ct_{i:06d}.seal"))
                batch.append(ct)
            p = cts_to_payload(batch, f"test_{start}")
            p["batch_start"] = start
            p["batch_total"] = N_TEST
            requests.post(SERVER_URL + "/api/upload/test",
                          json=p, timeout=120).raise_for_status()
            done = min(start + 100, N_TEST)
            if done % 500 == 0 or done == N_TEST:
                log(f"  Test re-upload: {done}/{N_TEST}")

        # Round 2 — feature sums
        log("Round 2 — Server computing feature sums...")
        update_state(message="Round 2 — Computing feature sums...")
        resp = requests.get(SERVER_URL + "/api/compute/sum",
                            timeout=120).json()
        ct_N = payload_to_single_ct(resp, context)

        # Round 3 — invert N
        log("Round 3 — Inverting N...")
        update_state(message="Round 3 — Inverting N...")
        val = decrypt_scalar(ct_N)
        log(f"  Decrypted N = {val:.0f}")
        ct_inv_N = invert_scalar(val)
        requests.post(SERVER_URL + "/api/invert/N",
                      json=single_ct_payload(ct_inv_N, "ct_inv_N"),
                      timeout=60).raise_for_status()

        # Round 4 — mu (and sigma² for SAD)
        log("Round 4 — Server computing mu...")
        update_state(message="Round 4 — Computing mu...")
        resp4 = requests.get(SERVER_URL + "/api/compute/mu",
                             timeout=120).json()

        if algorithm == "SAD":
            ct_sigma2 = payload_to_single_ct(resp4, context)
            log("Round 5 — Inverting sigma²...")
            update_state(message="Round 5 — Inverting sigma²...")
            values = decrypt_vector(ct_sigma2, N_FEATURES)
            log(f"  sigma² range [{values.min():.4f}, {values.max():.4f}]")
            ct_inv_s2 = invert_vector(values)
            requests.post(SERVER_URL + "/api/invert/sigma2",
                          json=single_ct_payload(ct_inv_s2, "ct_inv_sigma2"),
                          timeout=60).raise_for_status()

        # Trigger async score computation
        log("Triggering score computation (async)...")
        update_state(message="Server computing scores (async)...")
        requests.get(SERVER_URL + "/api/compute/scores",
                     timeout=30).raise_for_status()

        # Poll until ready
        t0 = time.time()
        while True:
            poll = requests.get(SERVER_URL + "/api/scores/status",
                                timeout=10).json()
            if poll["status"] == "ready":
                total = poll["count"]
                log(f"  {total} scores ready ✓")
                break
            elapsed = int(time.time() - t0)
            update_state(message=f"Server computing scores... ({elapsed}s)")
            if elapsed > SCORE_TIMEOUT:
                raise TimeoutError("Score computation timed out")
            time.sleep(5)

        # Fetch scores in batches
        log("Fetching scores in batches...")
        update_state(message="Fetching encrypted scores...")
        score_cts = []
        for start in range(0, total, 100):
            r = requests.get(
                f"{SERVER_URL}/api/scores/batch?start={start}&size=100",
                timeout=60)
            score_cts.extend(payload_to_cts(r.json(), context))
            done = min(start + 100, total)
            if done % 500 == 0 or done == total:
                log(f"  Fetched {done}/{total} scores")

        # Compute threshold from training rows
        log("Computing threshold from training rows...")
        update_state(message="Computing threshold...")
        df_train   = pd.read_csv(TRAIN_CSV)
        X_train    = df_train.values.astype(np.float64)[:N_TRAIN]
        slot_count = encoder.slot_count()
        encryptor_t, decryptor_t, _, _ = load_client_keys()
        train_scores = []

        if algorithm == "SAD":
            from server.algo import SAD as SadAlgo
            mu_vals      = X_train.mean(axis=0).tolist()
            sigma2_vals  = np.var(X_train, axis=0)
            inv_sigma2   = [1.0/max(float(v), 0.01) for v in sigma2_vals]
            mu_padded    = mu_vals + [0.0]*(slot_count - len(mu_vals))
            inv_s2_padded = inv_sigma2 + [0.0]*(slot_count - len(inv_sigma2))
            sad_local    = SadAlgo()
            for i in range(THRESHOLD_N):
                row    = X_train[i].tolist()
                padded = row + [0.0]*(slot_count - len(row))
                ct_i   = encryptor_t.encrypt(encoder.encode(padded, SCALE))
                sc     = sad_local.compute_scores_batch(
                    [ct_i], mu_padded, inv_s2_padded)
                dec    = encoder.decode(decryptor_t.decrypt(sc[0]))
                train_scores.append(float(np.real(dec[0])))
                if (i+1) % 100 == 0:
                    log(f"  Threshold: {i+1}/{THRESHOLD_N}")

        elif algorithm == "PCA":
            from server.pca_algo import PCA_HE
            from sklearn.decomposition import PCA as SklearnPCA
            mu_plaintext = X_train.mean(axis=0)
            pca_full     = SklearnPCA()
            pca_full.fit(X_train)
            cumvar  = np.cumsum(pca_full.explained_variance_ratio_)
            opt_k   = int(np.argmax(cumvar >= VARIANCE_THRESH) + 1)
            pca_fit = SklearnPCA(n_components=opt_k)
            pca_fit.fit(X_train)
            V       = pca_fit.components_
            pca_local = PCA_HE(n_components=opt_k)
            pt_mu   = mu_plaintext.tolist() + [0.0]*(slot_count - len(mu_plaintext))
            for i in range(THRESHOLD_N):
                row    = X_train[i].tolist()
                padded = row + [0.0]*(slot_count - len(row))
                ct_i   = encryptor_t.encrypt(encoder.encode(padded, SCALE))
                sc     = pca_local.compute_score(ct_i, mu_plaintext, V)
                dec    = encoder.decode(decryptor_t.decrypt(sc))
                train_scores.append(float(np.real(dec[0])))
                if (i+1) % 100 == 0:
                    log(f"  Threshold: {i+1}/{THRESHOLD_N}")

        threshold = float(np.mean(train_scores) + 3*np.std(train_scores))
        log(f"  Threshold = {threshold:.6f}")

        # Decrypt and decide
        log("Decrypting scores and classifying...")
        update_state(message="Decrypting and classifying...")
        predictions, scores = decide(score_cts, threshold,
                                     labels_path=LABELS_PATH)

        # Compute metrics and post to server
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

        requests.post(SERVER_URL + "/api/results", json={
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
        }, timeout=30)

        msg = f"Complete ✓  AUC={auc:.4f}  F1={f1:.4f}"
        log(msg)
        update_state(phase="done", message=msg)

    except Exception as e:
        update_state(phase="error", error=str(e),
                     message=f"Error: {e}")
        log(f"ERROR: {e}")

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html",
                           server_url=SERVER_URL,
                           sad_rounds=SAD_ROUNDS,
                           pca_rounds=PCA_ROUNDS)

# ─────────────────────────────────────────────
# SOCKETIO — button events from browser
# ─────────────────────────────────────────────
@socketio.on("action_upload")
def handle_upload(*args):
    if client_state["phase"] not in ("idle", "error"):
        socketio.emit("log", {"msg": "Already running — stop first"})
        return
    update_state(stop=False, paused=False)
    threading.Thread(target=_do_upload, daemon=True).start()

@socketio.on("action_pause")
def handle_pause(*args):
    with state_lock:
        client_state["paused"] = not client_state["paused"]
        msg = "Paused" if client_state["paused"] else "Resumed"
    socketio.emit("client_state", dict(client_state))
    socketio.emit("log", {"msg": msg})

@socketio.on("action_stop")
def handle_stop(*args):
    update_state(stop=True, paused=False,
                 phase="idle", message="Stopped — click Send Data to restart",
                 progress=0)
    socketio.emit("log", {"msg": "Stopped"})

@socketio.on("action_run_algorithm")
def handle_run(*args):
    data = args[0] if args and isinstance(args[0], dict) else {}
    algo = data.get("algorithm", "SAD").upper()
    if algo not in ("SAD", "PCA"):
        socketio.emit("log", {"msg": f"Unknown algorithm: {algo}"})
        return
    if client_state["phase"] == "running":
        socketio.emit("log", {"msg": "Already running"})
        return
    threading.Thread(target=_do_run_algorithm,
                     args=(algo,), daemon=True).start()

@socketio.on("action_reset")
def handle_reset(*args):
    update_state(phase="idle", progress=0, stop=False, paused=False,
                 algorithm=None, error=None,
                 message="Ready — click Send Data to begin")
    try:
        requests.post(SERVER_URL + "/api/reset",
                      json={"mode": "SAD"}, timeout=5)
    except Exception:
        pass
    socketio.emit("log", {"msg": "Reset ✓"})

if __name__ == "__main__":
    print(f"\nDashboard  →  http://localhost:{DASHBOARD_PORT}")
    print(f"HE Server  →  {SERVER_URL}\n")
    socketio.run(app, host="0.0.0.0", port=DASHBOARD_PORT, debug=False)