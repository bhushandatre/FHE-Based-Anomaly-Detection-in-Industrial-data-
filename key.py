"""
key.py
======
Sets up SEAL context and generates all encryption keys.

Saves:
    keys/server/  →  public_key.seal, relin_keys.seal, galois_keys.seal
    keys/client/  →  secret_key.seal, public_key.seal, relin_keys.seal, galois_keys.seal

Other files import context, encoder, evaluator, SCALE from here.
"""

import seal
from seal import (
    EncryptionParameters, scheme_type,
    SEALContext, KeyGenerator,
    Encryptor, Decryptor, Evaluator,
    CKKSEncoder, CoeffModulus,
    sec_level_type
)
import os

# ─────────────────────────────────────────────
# SEAL PARAMETERS
# poly_mod  = 16384
# coeff_mod = [60,40,40,40,40,40,60] → 5 usable levels
# scale     = 2^40
# slots     = 8192
# ─────────────────────────────────────────────
SCALE        = 2.0 ** 40
ROTATION_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

parms = EncryptionParameters(scheme_type.ckks)
parms.set_poly_modulus_degree(16384)
parms.set_coeff_modulus(CoeffModulus.Create(16384, [60, 40, 40, 40, 40, 40, 60]))

context    = SEALContext(parms, True, sec_level_type.tc128)
encoder    = CKKSEncoder(context)
evaluator  = Evaluator(context)
slot_count = encoder.slot_count()


def generate_keys(client_keys_dir="keys/client", server_keys_dir="keys/server"):
    """
    Generates all SEAL keys and saves them to client and server folders.

    Client gets : secret_key, public_key, relin_keys, galois_keys
    Server gets : public_key, relin_keys, galois_keys  (NO secret key)
    """
    os.makedirs(client_keys_dir, exist_ok=True)
    os.makedirs(server_keys_dir, exist_ok=True)

    print("[key.py] Generating keys...")

    # ── Generate ──────────────────────────────
    keygen      = KeyGenerator(context)
    secret_key  = keygen.secret_key()
    public_key  = keygen.create_public_key()
    relin_keys  = keygen.create_relin_keys()
    galois_keys = keygen.create_galois_keys()

    # ── Save to keys/client/ ──────────────────
    secret_key.save( os.path.join(client_keys_dir, "secret_key.seal"))
    public_key.save( os.path.join(client_keys_dir, "public_key.seal"))
    relin_keys.save( os.path.join(client_keys_dir, "relin_keys.seal"))
    galois_keys.save(os.path.join(client_keys_dir, "galois_keys.seal"))

    # ── Save to keys/server/ (no secret key) ─
    public_key.save( os.path.join(server_keys_dir, "public_key.seal"))
    relin_keys.save( os.path.join(server_keys_dir, "relin_keys.seal"))
    galois_keys.save(os.path.join(server_keys_dir, "galois_keys.seal"))

    print(f"  OK  keys/client/ → secret_key, public_key, relin_keys, galois_keys")
    print(f"  OK  keys/server/ → public_key, relin_keys, galois_keys (no secret key)")
    print(f"  OK  slots = {slot_count}, scale = 2^40, levels = 5")


# Base directory = folder where key.py lives (system/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_client_keys(client_keys_dir=None):
    """
    Loads all keys for client side.
    Returns: encryptor, decryptor, relin_keys, galois_keys
    """
    if client_keys_dir is None:
        client_keys_dir = os.path.join(BASE_DIR, "keys", "client")

    secret_key  = seal.SecretKey()
    public_key  = seal.PublicKey()
    relin_keys  = seal.RelinKeys()
    galois_keys = seal.GaloisKeys()

    secret_key.load( context, os.path.join(client_keys_dir, "secret_key.seal"))
    public_key.load( context, os.path.join(client_keys_dir, "public_key.seal"))
    relin_keys.load( context, os.path.join(client_keys_dir, "relin_keys.seal"))
    galois_keys.load(context, os.path.join(client_keys_dir, "galois_keys.seal"))

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)

    return encryptor, decryptor, relin_keys, galois_keys


def load_server_keys(server_keys_dir=None):
    """
    Loads keys for server side.
    Returns: encryptor, relin_keys, galois_keys
    Server has no secret key — cannot decrypt.
    """
    if server_keys_dir is None:
        server_keys_dir = os.path.join(BASE_DIR, "keys", "server")

    public_key  = seal.PublicKey()
    relin_keys  = seal.RelinKeys()
    galois_keys = seal.GaloisKeys()

    public_key.load( context, os.path.join(server_keys_dir, "public_key.seal"))
    relin_keys.load( context, os.path.join(server_keys_dir, "relin_keys.seal"))
    galois_keys.load(context, os.path.join(server_keys_dir, "galois_keys.seal"))

    encryptor = Encryptor(context, public_key)

    return encryptor, relin_keys, galois_keys


# ─────────────────────────────────────────────
# RUN DIRECTLY TO GENERATE KEYS
# python key.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("SEAL Key Generation")
    print("=" * 50)
    generate_keys()
    print("\n  Keys ready.")
 #   print("  from key import context, encoder, evaluator, SCALE")