"""
client/decrypt.py
=================
Decrypts a ciphertext to plaintext values.
Only the client can decrypt — client holds the secret key.

Usage:
    from client.decrypt import decrypt_values, decrypt_scalar
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from seal import Ciphertext
from key import context, encoder, load_client_keys


def decrypt_values(ct, n_values):
    """
    Decrypts a ciphertext and returns first n_values slots as numpy array.

    Args:
        ct       : seal.Ciphertext to decrypt
        n_values : how many slots to return

    Returns:
        numpy array of shape (n_values,) with real parts
    """
    _, decryptor, _, _ = load_client_keys()
    decoded = encoder.decode(decryptor.decrypt(ct))
    return np.real(np.array(decoded[:n_values], dtype=np.float64))


def decrypt_scalar(ct):
    """
    Decrypts a ciphertext and returns slot 0 as a single float.
    Used for decrypting scalar results like sigma² or anomaly score.

    Args:
        ct : seal.Ciphertext to decrypt

    Returns:
        float value from slot 0
    """
    _, decryptor, _, _ = load_client_keys()
    decoded = encoder.decode(decryptor.decrypt(ct))
    return float(np.real(decoded[0]))


def decrypt_vector(ct, n_features=51):
    """
    Decrypts a row-major ciphertext and returns feature values.
    Used for decrypting sigma² vector (one value per feature).

    Args:
        ct         : seal.Ciphertext to decrypt
        n_features : number of features (default 51)

    Returns:
        numpy array of shape (n_features,)
    """
    return decrypt_values(ct, n_features)