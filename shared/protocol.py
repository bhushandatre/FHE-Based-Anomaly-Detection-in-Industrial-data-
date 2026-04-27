"""
shared/protocol.py
==================
Handles serialization and deserialization of SEAL ciphertexts
for HTTP transfer between client and server.

Ciphertexts are serialized to base64 strings for JSON transport.
Both client and server use these helpers.
"""

import base64
import tempfile
import os
from seal import Ciphertext


def ct_to_base64(ct) -> str:
    """
    Serialize a SEAL ciphertext to a base64 string for HTTP transfer.

    Args:
        ct : seal.Ciphertext

    Returns:
        base64 encoded string
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".seal") as f:
        tmp_path = f.name
    try:
        ct.save(tmp_path)
        with open(tmp_path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def base64_to_ct(b64_str: str, context) -> Ciphertext:
    """
    Deserialize a base64 string back to a SEAL ciphertext.

    Args:
        b64_str : base64 encoded string
        context : SEAL context

    Returns:
        seal.Ciphertext
    """
    raw = base64.b64decode(b64_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".seal") as f:
        tmp_path = f.name
        f.write(raw)
    try:
        ct = Ciphertext()
        ct.load(context, tmp_path)
        return ct
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def cts_to_payload(ct_list: list, label: str = "") -> dict:
    """
    Serialize a list of ciphertexts into a JSON-ready dict.

    Args:
        ct_list : list of seal.Ciphertext
        label   : optional label (e.g. "train", "test", "inv_N")

    Returns:
        dict with keys: label, count, ciphertexts (list of b64 strings)
    """
    return {
        "label"      : label,
        "count"      : len(ct_list),
        "ciphertexts": [ct_to_base64(ct) for ct in ct_list]
    }


def payload_to_cts(payload: dict, context) -> list:
    """
    Deserialize a JSON payload back to a list of ciphertexts.

    Args:
        payload : dict from cts_to_payload
        context : SEAL context

    Returns:
        list of seal.Ciphertext
    """
    return [base64_to_ct(b64, context) for b64 in payload["ciphertexts"]]


def single_ct_payload(ct, label: str = "") -> dict:
    """Serialize a single ciphertext."""
    return {
        "label"     : label,
        "ciphertext": ct_to_base64(ct)
    }


def payload_to_single_ct(payload: dict, context) -> Ciphertext:
    """Deserialize a single ciphertext payload."""
    return base64_to_ct(payload["ciphertext"], context)