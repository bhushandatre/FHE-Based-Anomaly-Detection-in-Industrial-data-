"""
server/messenger.py
===================
Handles all message sending and receiving on the server side.

Responsibilities:
    - Send INVERT_REQUEST to client (for N and sigma²)
    - Receive INVERT_RESPONSE from client
    - Send SCORE to client

In simulation (run.py), messages are passed as objects.
In real deployment, these would be sent over HTTP/socket.

Usage:
    from server.messenger import ServerMessenger
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from message import (
    MessageType, InvertContext,
    InvertRequestMessage, ScoreMessage,
    MessageParser
)
from key import context
from seal import Ciphertext


class ServerMessenger:
    def __init__(self):
        self.pending_response = None

    def send_invert_request(self, ct_list, invert_context):
        """
        Packages encrypted denominator(s) into an INVERT_REQUEST message.
        In simulation, returns the message object directly.
        In real deployment, this would send over network.

        Args:
            ct_list        : list of seal.Ciphertext to be inverted
            invert_context : InvertContext.N or InvertContext.SIGMA2

        Returns:
            InvertRequestMessage object
        """
        msg = InvertRequestMessage.from_ciphertexts(invert_context, ct_list)
        print(f"[server/messenger.py] Sending INVERT_REQUEST "
              f"(context={invert_context}, n_cts={len(ct_list)})")
        return msg

    def receive_invert_response(self, msg):
        """
        Receives and unpacks INVERT_RESPONSE from client.

        Args:
            msg : InvertResponseMessage object

        Returns:
            list of seal.Ciphertext (inverted values)
        """
        ct_list = msg.to_ciphertexts(Ciphertext, context)
        print(f"[server/messenger.py] Received INVERT_RESPONSE "
              f"(context={msg.context}, n_cts={len(ct_list)})")
        return ct_list

    def send_scores(self, score_ct_list):
        """
        Packages encrypted scores into a SCORE message.

        Args:
            score_ct_list : list of seal.Ciphertext (one score per test row)

        Returns:
            ScoreMessage object
        """
        msg = ScoreMessage.from_ciphertexts(score_ct_list, len(score_ct_list))
        print(f"[server/messenger.py] Sending SCORE "
              f"(n_rows={len(score_ct_list)})")
        return msg