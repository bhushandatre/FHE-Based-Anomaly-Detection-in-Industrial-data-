"""
client/invert.py
================
Receives plaintext values from decrypt.py,
computes 1/value (exact plaintext division),
re-encrypts and returns ciphertext.

This is the core of the multi-round protocol —
division is done here in plaintext, not inside HE.

Usage:
    from client.invert import invert_scalar, invert_vector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from key import context, encoder, SCALE, load_client_keys


def invert_scalar(value):
    """
    Computes 1/value and returns encrypted result.
    Used for inverting N (count) to get 1/N for mean computation.

    Args:
        value : plaintext scalar (e.g. N = 5000)

    Returns:
        seal.Ciphertext containing 1/value in all slots
    """
    encryptor, _, _, _ = load_client_keys()
    slot_count         = encoder.slot_count()

    inv_value  = 1.0 / float(value)
    padded     = [inv_value] * slot_count
    plain      = encoder.encode(padded, SCALE)
    ct         = encryptor.encrypt(plain)

    return ct


def invert_vector(values):
    """
    Computes 1/values elementwise and returns encrypted result.
    Used for inverting sigma² vector (one value per feature).

    Args:
        values : numpy array of shape (n_features,)
                 e.g. sigma² values for all 51 features

    Returns:
        seal.Ciphertext where slot j = 1/values[j]
    """
    encryptor, _, _, _ = load_client_keys()
    slot_count         = encoder.slot_count()

    # Clamp to avoid division by zero
    values     = np.array(values, dtype=np.float64)
    values     = np.where(np.abs(values) < 1e-8, 1e-8, values)

    inv_values = 1.0 / values
    padded     = inv_values.tolist() + [0.0] * (slot_count - len(inv_values))
    plain      = encoder.encode(padded, SCALE)
    ct         = encryptor.encrypt(plain)

    return ct