"""
shared/config.py
================
Single source of truth for all configuration.
Both client and server import from here.
Change SERVER_HOST to deploy on different machine or cloud.
"""

# ─────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────
SERVER_HOST      = "127.0.0.1"     # change to server IP for cloud deployment
SERVER_PORT      = 8000            # FastAPI server
DASHBOARD_PORT   = 3000            # Flask dashboard
SERVER_URL       = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ─────────────────────────────────────────────
# TIMEOUTS
# ─────────────────────────────────────────────
REQUEST_TIMEOUT  = 300    # seconds — default timeout
SCORE_TIMEOUT    = 3600   # seconds — score computation can take very long for PCA
POLL_INTERVAL    = 2.0    # seconds — how often client polls server
MAX_RETRIES      = 5      # retries on network failure

# ─────────────────────────────────────────────
# SEAL PARAMETERS
# ─────────────────────────────────────────────
POLY_MOD_DEGREE  = 16384
COEFF_MOD_BITS   = [60, 40, 40, 40, 40, 40, 60]
SCALE_BITS       = 40
N_SLOTS          = 8192   # poly_mod / 2

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
N_TRAIN          = 5000
N_TEST           = 2000
N_FEATURES       = 51
N_COMPONENTS     = None   # determined dynamically from VARIANCE_THRESH
VARIANCE_THRESH  = 0.90   # keep components explaining 90% of variance
THRESHOLD_N      = 500    # training rows for threshold computation
ATTACK_RATIO     = 0.10

# ─────────────────────────────────────────────
# PATHS (relative to system/ folder)
# ─────────────────────────────────────────────
TRAIN_CSV        = "data/normal_preprocessed.csv"
TEST_CSV         = "data/swat_unified_dataset.csv"
TRAIN_CT_DIR     = "encrypted/train"
TEST_CT_DIR      = "encrypted/test"
LABELS_PATH      = "encrypted/test_labels.npy"
CLIENT_KEYS_DIR  = "keys/client"
SERVER_KEYS_DIR  = "keys/server"

# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────
ENDPOINTS = {
    # SAD protocol
    "upload_train"    : "/api/upload/train",
    "compute_sum"     : "/api/compute/sum",
    "invert_N"        : "/api/invert/N",
    "compute_mu"      : "/api/compute/mu",
    "invert_sigma2"   : "/api/invert/sigma2",
    "upload_test"     : "/api/upload/test",
    "compute_scores"  : "/api/compute/scores",

    # PCA protocol
    "compute_mu_pca"  : "/api/pca/compute/mu",
    "compute_scores_pca": "/api/pca/compute/scores",

    # Status and results
    "status"          : "/api/status",
    "results"         : "/api/results",
    "reset"           : "/api/reset",
    "health"          : "/api/health",
}

# ─────────────────────────────────────────────
# PROTOCOL ROUNDS (for dashboard display)
# ─────────────────────────────────────────────
SAD_ROUNDS = [
    "Round 1 — Client sends training data",
    "Round 2 — Server computes feature sums",
    "Round 3 — Client inverts N",
    "Round 4 — Server computes mu and sigma²",
    "Round 5 — Client inverts sigma²",
    "Round 6 — Client sends test data",
    "Round 7 — Server computes SAD scores",
    "Round 8 — Server sends scores to client",
    "Round 9 — Client decrypts and decides",
]

PCA_ROUNDS = [
    "Round 1 — Client sends training data",
    "Round 2 — Server computes feature sums",
    "Round 3 — Client inverts N",
    "Round 4 — Server computes mu and fits PCA",
    "Round 5 — Client sends test data",
    "Round 6 — Server computes PCA scores",
    "Round 7 — Server sends scores to client",
    "Round 8 — Client decrypts and decides",
]