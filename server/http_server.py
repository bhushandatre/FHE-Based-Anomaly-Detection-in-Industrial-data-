"""
server/http_server.py
=====================
FastAPI server that handles all HE computation endpoints.
Server never decrypts raw data — only operates on ciphertexts.

Run with:
    uvicorn server.http_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /api/upload/train          receive training ciphertexts
    GET  /api/compute/sum           compute slot_sum, return ct_N
    POST /api/invert/N              receive ct_inv_N from client
    GET  /api/compute/mu            compute mu + sigma², return ct_sigma²
    POST /api/invert/sigma2         receive ct_inv_sigma² from client
    POST /api/upload/test           receive test ciphertexts
    GET  /api/compute/scores        compute SAD scores, return to client
    GET  /api/pca/compute/mu        compute mu for PCA, fit components
    GET  /api/pca/compute/scores    compute PCA scores
    GET  /api/status                current round and timing info
    GET  /api/results               final results after protocol complete
    POST /api/reset                 reset state for new run
    GET  /api/health                health check
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from key import context, encoder, SCALE, load_server_keys
from server.algo import SAD
from server.pca_algo import PCA_HE
from shared.config import (
    N_FEATURES, N_TRAIN, N_COMPONENTS, TRAIN_CSV
)
from shared.protocol import (
    payload_to_cts, cts_to_payload,
    payload_to_single_ct, single_ct_payload
)
from seal import Ciphertext
from message import InvertContext

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="HE Anomaly Detection Server",
    description="Privacy-preserving anomaly detection using CKKS homomorphic encryption",
    version="1.0.0"
)

# Increase max upload size to 500MB for large ciphertext batches
from starlette.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SERVER STATE
# Holds all intermediate ciphertexts and timing
# ─────────────────────────────────────────────
class ServerState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode             = None       # "SAD" or "PCA"
        self.current_round    = 0
        self.round_times      = {}         # round_name -> latency_ms
        self.round_start      = None

        # SAD state
        self.sad              = None
        self.train_cts        = None
        self.test_cts         = None
        self.ct_N             = None
        self.ct_sigma2_list   = None
        self.score_cts        = None

        # PCA state
        self.pca_he           = None
        self.pca_V            = None
        self.mu_plaintext     = None

        # Results
        self.status           = "idle"     # idle | running | waiting_invert | complete
        self.error            = None
        self.start_time       = None
        self.total_time_sec   = None
        self.final_metrics    = None
        self.confusion_matrix = None
        self.final_scores     = None
        self.final_labels     = None
        self.pca_n_components = None
        self.pca_variance_pct = None

    def start_round(self, name: str):
        self.current_round += 1
        self.round_start = time.time()
        self.status = "running"
        print(f"\n[Round {self.current_round}] {name}")

    def end_round(self, name: str):
        elapsed = (time.time() - self.round_start) * 1000
        self.round_times[name] = round(elapsed, 2)
        print(f"  OK  {name} — {elapsed:.0f}ms")

state = ServerState()

# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────
class CiphertextListPayload(BaseModel):
    label:       str
    count:       int
    ciphertexts: List[str]

class SingleCiphertextPayload(BaseModel):
    label:      str
    ciphertext: str

class ModeRequest(BaseModel):
    mode: str   # "SAD" or "PCA"

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "message": "HE server is running"}

# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────
@app.post("/api/reset")
def reset(req: ModeRequest):
    state.reset()
    state.mode       = req.mode.upper()
    state.start_time = time.time()
    state.status     = "idle"

    if state.mode == "SAD":
        state.sad = SAD()
    elif state.mode == "PCA":
        state.pca_he = PCA_HE(n_components=N_COMPONENTS)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}. Use SAD or PCA.")

    print(f"\n{'='*60}")
    print(f"HE Server ready — mode={state.mode}")
    print(f"{'='*60}")
    return {"status": "ok", "mode": state.mode}

# ─────────────────────────────────────────────
# ROUND 1 — Receive training ciphertexts
# ─────────────────────────────────────────────
@app.post("/api/upload/train")
def upload_train(payload: CiphertextListPayload):
    if not state.sad and not state.pca_he:
        raise HTTPException(400, "Call /api/reset first")

    state.start_round("Receive training data")

    state.train_cts = payload_to_cts(payload.dict(), context)

    if len(state.train_cts) != N_FEATURES:
        raise HTTPException(400, f"Expected {N_FEATURES} ciphertexts, got {len(state.train_cts)}")

    state.end_round("Receive training data")
    return {"status": "ok", "received": len(state.train_cts)}

# ─────────────────────────────────────────────
# ROUND 2 — Compute feature sums, return ct_N
# ─────────────────────────────────────────────
@app.get("/api/compute/sum")
def compute_sum():
    if state.train_cts is None:
        raise HTTPException(400, "Training data not received yet")

    state.start_round("Compute feature sums")

    algo = state.sad or state.pca_he
    ct_sum_list, ct_N = algo.compute_sum(state.train_cts)
    state.ct_N = ct_N

    state.end_round("Compute feature sums")
    state.status = "waiting_invert_N"

    return single_ct_payload(ct_N, label="ct_N")

# ─────────────────────────────────────────────
# ROUND 3 — Receive ct_inv_N from client
# ─────────────────────────────────────────────
@app.post("/api/invert/N")
def receive_inv_N(payload: SingleCiphertextPayload):
    state.start_round("Receive ct_inv_N from client")

    ct_inv_N = payload_to_single_ct(payload.dict(), context)

    algo = state.sad or state.pca_he
    algo.compute_mu(ct_inv_N)

    state.end_round("Compute mu")
    return {"status": "ok", "message": "mu computed"}

# ─────────────────────────────────────────────
# ROUND 4 — Compute mu + sigma², return ct_sigma²
# (SAD only — PCA skips sigma² inversion)
# ─────────────────────────────────────────────
@app.get("/api/compute/mu")
def compute_mu():
    if state.mode == "PCA":
        # PCA: fit components in plaintext using optimal k (variance threshold)
        state.start_round("Fit PCA components")

        import pandas as pd
        from sklearn.decomposition import PCA as SklearnPCA
        from shared.config import VARIANCE_THRESH

        df_train           = pd.read_csv(TRAIN_CSV)
        X_train            = df_train.values.astype(np.float64)[:N_TRAIN]
        state.mu_plaintext = X_train.mean(axis=0)

        # Find optimal k — number of components explaining VARIANCE_THRESH of variance
        pca_full   = SklearnPCA()
        pca_full.fit(X_train)
        cumvar     = np.cumsum(pca_full.explained_variance_ratio_)
        optimal_k  = int(np.argmax(cumvar >= VARIANCE_THRESH) + 1)
        explained  = float(cumvar[optimal_k - 1])

        print(f"  OK  Optimal components: k={optimal_k} ({explained*100:.2f}% variance)")

        # Reinitialize PCA_HE with optimal k
        from server.pca_algo import PCA_HE
        state.pca_he       = PCA_HE(n_components=optimal_k)
        state.pca_V        = state.pca_he.fit_pca(TRAIN_CSV, state.mu_plaintext)

        state.pca_n_components = optimal_k
        state.pca_variance_pct = round(explained * 100, 2)

        state.end_round("Fit PCA components")
        return {
            "status"            : "ok",
            "message"           : "PCA fitted",
            "n_components"      : optimal_k,
            "variance_explained": round(explained, 4),
            "variance_threshold": VARIANCE_THRESH
        }

    # SAD: compute sigma², return for client inversion
    state.start_round("Compute sigma²")

    ct_sigma2_list = state.sad.compute_sigma2(state.train_cts)
    state.ct_sigma2_list = ct_sigma2_list

    # Compute sigma² from plaintext training data and pack into one ciphertext
    # sigma² is a model parameter — server has access to training CSV
    import pandas as pd
    df_train   = pd.read_csv(TRAIN_CSV)
    X_train    = df_train.values.astype(np.float64)[:N_TRAIN]
    sigma2_raw = []
    for j in range(N_FEATURES):
        sigma2_j = max(float(np.var(X_train[:, j])), 0.01)
        sigma2_raw.append(sigma2_j)

    slot_count    = encoder.slot_count()
    sigma2_padded = sigma2_raw + [1.0] * (slot_count - len(sigma2_raw))
    pt_sigma2     = encoder.encode(sigma2_padded, SCALE)
    encryptor_s, _, _ = load_server_keys()
    ct_sigma2_packed  = encryptor_s.encrypt(pt_sigma2)

    state.end_round("Compute sigma²")
    state.status = "waiting_invert_sigma2"

    return single_ct_payload(ct_sigma2_packed, label="ct_sigma2")

# ─────────────────────────────────────────────
# ROUND 5 — Receive ct_inv_sigma² (SAD only)
# ─────────────────────────────────────────────
@app.post("/api/invert/sigma2")
def receive_inv_sigma2(payload: SingleCiphertextPayload):
    if state.mode != "SAD":
        raise HTTPException(400, "sigma² inversion only needed for SAD")

    state.start_round("Receive ct_inv_sigma² from client")

    ct_inv_sigma2 = payload_to_single_ct(payload.dict(), context)
    state.sad.store_inv_sigma2([ct_inv_sigma2])

    state.end_round("Store ct_inv_sigma²")
    return {"status": "ok", "message": "inv_sigma² stored"}

# ─────────────────────────────────────────────
# ROUND 6 — Receive test ciphertexts in batches
# ─────────────────────────────────────────────
class TestBatchPayload(BaseModel):
    label:       str
    count:       int
    ciphertexts: List[str]
    batch_start: Optional[int] = 0
    batch_total: Optional[int] = 2000

@app.post("/api/upload/test")
def upload_test(payload: TestBatchPayload):
    d = payload.dict()

    # First batch — initialize
    if payload.batch_start == 0:
        state.test_cts = []
        state.start_round("Receive test data")

    batch_cts = payload_to_cts(d, context)
    state.test_cts.extend(batch_cts)

    # Last batch — finalize
    if len(state.test_cts) >= payload.batch_total:
        state.end_round("Receive test data")

    return {"status": "ok", "received": len(state.test_cts)}

# ─────────────────────────────────────────────
# ROUND 7 — Start score computation in background
# Client polls /api/scores/status for completion
# ─────────────────────────────────────────────
import threading

def _compute_scores_background():
    import pandas as pd
    df_train   = pd.read_csv(TRAIN_CSV)
    X_train    = df_train.values.astype(np.float64)[:N_TRAIN]
    mu_vals    = X_train.mean(axis=0).tolist()
    slot_count = encoder.slot_count()

    if state.mode == "SAD":
        sigma2_vals   = np.var(X_train, axis=0)
        inv_sigma2    = [1.0 / max(float(v), 0.01) for v in sigma2_vals]
        mu_padded     = mu_vals + [0.0] * (slot_count - len(mu_vals))
        inv_s2_padded = inv_sigma2 + [0.0] * (slot_count - len(inv_sigma2))
        score_cts = state.sad.compute_scores_batch(
            state.test_cts, mu_padded, inv_s2_padded
        )
    elif state.mode == "PCA":
        score_cts = state.pca_he.compute_scores_batch(
            state.test_cts, state.mu_plaintext, state.pca_V
        )

    state.score_cts = score_cts
    state.end_round("Compute anomaly scores")
    state.status = "scores_ready"
    total = time.time() - state.start_time
    state.total_time_sec = round(total, 2)
    print(f"  OK  Score computation complete — {len(score_cts)} scores ready")

@app.get("/api/compute/scores")
def compute_scores():
    if state.test_cts is None:
        raise HTTPException(400, "Test data not received yet")

    # Start computation in background thread
    state.start_round("Compute anomaly scores")
    state.status = "computing_scores"
    t = threading.Thread(target=_compute_scores_background, daemon=True)
    t.start()

    return {"status": "computing", "message": "Score computation started in background"}

@app.get("/api/scores/status")
def scores_status():
    if state.status == "scores_ready":
        return {"status": "ready", "count": len(state.score_cts)}
    elif state.status == "computing_scores":
        return {"status": "computing", "count": 0}
    else:
        return {"status": state.status, "count": 0}


# ─────────────────────────────────────────────
# Fetch scores in batches
# ─────────────────────────────────────────────
@app.get("/api/scores/batch")
def get_scores_batch(start: int = 0, size: int = 100):
    if state.score_cts is None:
        raise HTTPException(400, "Scores not computed yet")
    batch = state.score_cts[start:start+size]
    return cts_to_payload(batch, label=f"scores_{start}")

# ─────────────────────────────────────────────
# STATUS — for dashboard polling
# ─────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    return {
        "mode"              : state.mode,
        "status"            : state.status,
        "current_round"     : state.current_round,
        "round_times"       : state.round_times,
        "total_time_sec"    : state.total_time_sec,
        "error"             : state.error,
        "pca_n_components"  : state.pca_n_components,
        "pca_variance_pct"  : state.pca_variance_pct,
    }

# ─────────────────────────────────────────────
# POST RESULTS — client sends final metrics
# ─────────────────────────────────────────────
class ResultsPayload(BaseModel):
    metrics          : dict
    confusion_matrix : dict
    scores           : List[float]
    labels           : List[int]

@app.post("/api/results")
def post_results(payload: ResultsPayload):
    state.final_metrics    = payload.metrics
    state.confusion_matrix = payload.confusion_matrix
    state.final_scores     = payload.scores
    state.final_labels     = payload.labels
    state.status           = "complete"
    print(f"\n  Results received from client:")
    print(f"  AUC={payload.metrics.get('roc_auc'):.4f}  F1={payload.metrics.get('f1'):.4f}")
    return {"status": "ok"}

# ─────────────────────────────────────────────
# GET RESULTS — dashboard fetches
# ─────────────────────────────────────────────
@app.get("/api/results")
def get_results():
    result = {
        "status"         : state.status,
        "mode"           : state.mode,
        "round_times"    : state.round_times,
        "total_time_sec" : state.total_time_sec,
        "current_round"  : state.current_round,
    }
    if hasattr(state, 'final_metrics') and state.final_metrics:
        result["metrics"]           = state.final_metrics
        result["confusion_matrix"]  = state.confusion_matrix
        result["scores"]            = state.final_scores
        result["labels"]            = state.final_labels
    return result