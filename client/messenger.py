"""
client/messenger.py
===================
Handles all message sending and receiving on the client side.

Responsibilities:
    - Send encrypted training data to server
    - Send encrypted test data to server
    - Receive INVERT_REQUEST from server
    - Call decrypt.py and invert.py to process the request
    - Send INVERT_RESPONSE back to server
    - Receive SCORE from server

Usage:
    from client.messenger import ClientMessenger
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from message import (
    MessageType, InvertContext,
    DataMessage, InvertResponseMessage,
    MessageParser
)
from key import context
from seal import Ciphertext
from client.decrypt import decrypt_scalar, decrypt_vector
from client.invert  import invert_scalar, invert_vector


class ClientMessenger:

    def send_train_data(self, train_ct_list):
        """
        Packages encrypted training ciphertexts into a DATA message.

        Args:
            train_ct_list : list of 51 seal.Ciphertext (column-major)

        Returns:
            DataMessage object
        """
        msg = DataMessage.from_ciphertexts("TRAIN", train_ct_list)
        print(f"[client/messenger.py] Sending TRAIN DATA "
              f"(n_cts={len(train_ct_list)})")
        return msg

    def send_test_data(self, test_ct_list):
        """
        Packages encrypted test ciphertexts into a DATA message.

        Args:
            test_ct_list : list of N_TEST seal.Ciphertext (row-major)

        Returns:
            DataMessage object
        """
        msg = DataMessage.from_ciphertexts("TEST", test_ct_list)
        print(f"[client/messenger.py] Sending TEST DATA "
              f"(n_cts={len(test_ct_list)})")
        return msg

    def handle_invert_request(self, invert_request_msg):
        """
        Receives INVERT_REQUEST, processes it, returns INVERT_RESPONSE.

        Steps:
            1. Deserialize ciphertexts from message
            2. Decrypt each ciphertext → plaintext value
            3. Compute 1/value in plaintext
            4. Re-encrypt result
            5. Return as INVERT_RESPONSE message

        Args:
            invert_request_msg : InvertRequestMessage object

        Returns:
            InvertResponseMessage object
        """
        context_type = invert_request_msg.context
        ct_list      = invert_request_msg.to_ciphertexts(Ciphertext, context)

        print(f"[client/messenger.py] Handling INVERT_REQUEST "
              f"(context={context_type})")

        if context_type == InvertContext.N:
            # Single scalar — invert N to get 1/N
            value   = decrypt_scalar(ct_list[0])
            print(f"  OK  Decrypted N = {value:.0f}")
            ct_inv  = invert_scalar(value)
            ct_inv_list = [ct_inv]

        elif context_type == InvertContext.SIGMA2:
            # Vector of 51 sigma² values — invert each
            values      = decrypt_vector(ct_list[0], n_features=51)
            print(f"  OK  Decrypted sigma² range: "
                  f"[{values.min():.6f}, {values.max():.6f}]")
            ct_inv      = invert_vector(values)
            ct_inv_list = [ct_inv]

        else:
            raise ValueError(f"Unknown invert context: {context_type}")

        msg = InvertResponseMessage.from_ciphertexts(context_type, ct_inv_list)
        print(f"  OK  Sent INVERT_RESPONSE (context={context_type})")
        return msg

    def receive_scores(self, score_msg):
        """
        Receives SCORE message and unpacks ciphertexts.

        Args:
            score_msg : ScoreMessage object

        Returns:
            list of seal.Ciphertext (encrypted scores)
        """
        ct_list = score_msg.to_ciphertexts(Ciphertext, context)
        print(f"[client/messenger.py] Received SCORE "
              f"(n_rows={len(ct_list)})")
        return ct_list