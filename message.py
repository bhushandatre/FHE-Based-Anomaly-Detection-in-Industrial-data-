"""
message.py
============================================================
Message format for Multi-Round HE Protocol
============================================================

Two parties:
    Client  — has secret key, encrypts/decrypts data
    Cloud   — has public key only, computes on ciphertexts

Message Types:
    DATA             client → cloud   encrypted sensor data
    INVERT_REQUEST   cloud  → client  please invert this encrypted value
    INVERT_RESPONSE  client → cloud   here is the inverted value
    SCORE            cloud  → client  here is the encrypted anomaly score
"""

import json
import base64
import os


# ─────────────────────────────────────────────
# MESSAGE TYPES
# ─────────────────────────────────────────────
class MessageType:
    DATA             = "DATA"
    INVERT_REQUEST   = "INVERT_REQUEST"
    INVERT_RESPONSE  = "INVERT_RESPONSE"
    SCORE            = "SCORE"


# ─────────────────────────────────────────────
# INVERT CONTEXT
# Tells client WHAT is being inverted
# so client knows what it is computing
# ─────────────────────────────────────────────
class InvertContext:
    N       = "N"        # inverting count N → gives 1/N for mean computation
    SIGMA2  = "SIGMA2"   # inverting sigma² → gives 1/sigma² for SAD score


# ─────────────────────────────────────────────
# BASE MESSAGE
# ─────────────────────────────────────────────
class Message:
    def __init__(self, msg_type):
        self.msg_type = msg_type

    def to_dict(self):
        return {"type": self.msg_type}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def ciphertext_to_bytes(ct, path="/tmp/he_msg_temp.seal"):
        """Save ciphertext to bytes via temp file."""
        ct.save(path)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def bytes_to_ciphertext(ct_class, context, b64_str,
                            path="/tmp/he_msg_temp.seal"):
        """Load ciphertext from bytes via temp file."""
        data = base64.b64decode(b64_str)
        with open(path, "wb") as f:
            f.write(data)
        ct = ct_class()
        ct.load(context, path)
        os.remove(path)
        return ct


# ─────────────────────────────────────────────
# MESSAGE 1: DATA
# Client → Cloud
# Sends encrypted training or test ciphertexts
#
# For training: column-major (51 ciphertexts, one per feature)
# For testing:  row-major    (one ciphertext per test row)
# ─────────────────────────────────────────────
class DataMessage(Message):
    def __init__(self, data_type, ciphertexts):
        """
        data_type    : "TRAIN" or "TEST"
        ciphertexts  : list of ciphertext byte strings (base64 encoded)
        """
        super().__init__(MessageType.DATA)
        self.data_type   = data_type
        self.ciphertexts = ciphertexts

    def to_dict(self):
        return {
            "type"        : self.msg_type,
            "data_type"   : self.data_type,
            "n_ciphertexts": len(self.ciphertexts),
            "ciphertexts" : self.ciphertexts
        }

    @classmethod
    def from_ciphertexts(cls, data_type, ct_list):
        """
        Build DataMessage from list of seal.Ciphertext objects.
        Serializes each to base64 string.
        """
        ct_bytes = []
        for i, ct in enumerate(ct_list):
            path = f"/tmp/he_msg_ct_{i}.seal"
            ct.save(path)
            with open(path, "rb") as f:
                ct_bytes.append(base64.b64encode(f.read()).decode("utf-8"))
            os.remove(path)
        return cls(data_type, ct_bytes)

    def to_ciphertexts(self, ct_class, context):
        """
        Deserialize back to list of seal.Ciphertext objects.
        """
        ct_list = []
        for i, b64 in enumerate(self.ciphertexts):
            path = f"/tmp/he_msg_ct_{i}.seal"
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            ct = ct_class()
            ct.load(context, path)
            os.remove(path)
            ct_list.append(ct)
        return ct_list

    @classmethod
    def from_dict(cls, d):
        return cls(d["data_type"], d["ciphertexts"])


# ─────────────────────────────────────────────
# MESSAGE 2: INVERT_REQUEST
# Cloud → Client
# Cloud sends an encrypted denominator and asks
# client to decrypt, invert, and return encrypted 1/denominator
#
# context tells client WHAT is being inverted:
#   N      → client will compute 1/N
#   SIGMA2 → client will compute 1/sigma² for each feature
# ─────────────────────────────────────────────
class InvertRequestMessage(Message):
    def __init__(self, context, ciphertexts):
        """
        context     : InvertContext.N or InvertContext.SIGMA2
        ciphertexts : list of base64 encoded ciphertext strings
                      For N:      1 ciphertext  (scalar N)
                      For SIGMA2: 51 ciphertexts (one per feature)
        """
        super().__init__(MessageType.INVERT_REQUEST)
        self.context     = context
        self.ciphertexts = ciphertexts

    def to_dict(self):
        return {
            "type"         : self.msg_type,
            "context"      : self.context,
            "n_ciphertexts": len(self.ciphertexts),
            "ciphertexts"  : self.ciphertexts
        }

    @classmethod
    def from_ciphertexts(cls, context, ct_list):
        """Build from list of seal.Ciphertext objects."""
        ct_bytes = []
        for i, ct in enumerate(ct_list):
            path = f"/tmp/he_inv_req_{i}.seal"
            ct.save(path)
            with open(path, "rb") as f:
                ct_bytes.append(base64.b64encode(f.read()).decode("utf-8"))
            os.remove(path)
        return cls(context, ct_bytes)

    def to_ciphertexts(self, ct_class, context):
        """Deserialize to list of seal.Ciphertext objects."""
        ct_list = []
        for i, b64 in enumerate(self.ciphertexts):
            path = f"/tmp/he_inv_req_{i}.seal"
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            ct = ct_class()
            ct.load(context, path)
            os.remove(path)
            ct_list.append(ct)
        return ct_list

    @classmethod
    def from_dict(cls, d):
        return cls(d["context"], d["ciphertexts"])


# ─────────────────────────────────────────────
# MESSAGE 3: INVERT_RESPONSE
# Client → Cloud
# Client returns encrypted 1/denominator
#
# Client:
#   1. Decrypts received ciphertexts
#   2. Computes 1/value in plaintext (exact division)
#   3. Encrypts result
#   4. Sends back as INVERT_RESPONSE
# ─────────────────────────────────────────────
class InvertResponseMessage(Message):
    def __init__(self, context, ciphertexts):
        """
        context     : same context as the request (N or SIGMA2)
        ciphertexts : list of base64 encoded ciphertext strings
                      containing encrypted 1/denominator values
        """
        super().__init__(MessageType.INVERT_RESPONSE)
        self.context     = context
        self.ciphertexts = ciphertexts

    def to_dict(self):
        return {
            "type"         : self.msg_type,
            "context"      : self.context,
            "n_ciphertexts": len(self.ciphertexts),
            "ciphertexts"  : self.ciphertexts
        }

    @classmethod
    def from_ciphertexts(cls, context, ct_list):
        """Build from list of seal.Ciphertext objects."""
        ct_bytes = []
        for i, ct in enumerate(ct_list):
            path = f"/tmp/he_inv_resp_{i}.seal"
            ct.save(path)
            with open(path, "rb") as f:
                ct_bytes.append(base64.b64encode(f.read()).decode("utf-8"))
            os.remove(path)
        return cls(context, ct_bytes)

    def to_ciphertexts(self, ct_class, context):
        """Deserialize to list of seal.Ciphertext objects."""
        ct_list = []
        for i, b64 in enumerate(self.ciphertexts):
            path = f"/tmp/he_inv_resp_{i}.seal"
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            ct = ct_class()
            ct.load(context, path)
            os.remove(path)
            ct_list.append(ct)
        return ct_list

    @classmethod
    def from_dict(cls, d):
        return cls(d["context"], d["ciphertexts"])


# ─────────────────────────────────────────────
# MESSAGE 4: SCORE
# Cloud → Client
# Cloud sends encrypted anomaly scores
# Client decrypts and decides Normal or Attack
# ─────────────────────────────────────────────
class ScoreMessage(Message):
    def __init__(self, ciphertexts, n_rows):
        """
        ciphertexts : list of base64 encoded score ciphertexts
        n_rows      : number of test rows scored
        """
        super().__init__(MessageType.SCORE)
        self.ciphertexts = ciphertexts
        self.n_rows      = n_rows

    def to_dict(self):
        return {
            "type"        : self.msg_type,
            "n_rows"      : self.n_rows,
            "n_ciphertexts": len(self.ciphertexts),
            "ciphertexts" : self.ciphertexts
        }

    @classmethod
    def from_ciphertexts(cls, ct_list, n_rows):
        """Build from list of seal.Ciphertext objects."""
        ct_bytes = []
        for i, ct in enumerate(ct_list):
            path = f"/tmp/he_score_{i}.seal"
            ct.save(path)
            with open(path, "rb") as f:
                ct_bytes.append(base64.b64encode(f.read()).decode("utf-8"))
            os.remove(path)
        return cls(ct_bytes, n_rows)

    def to_ciphertexts(self, ct_class, context):
        """Deserialize to list of seal.Ciphertext objects."""
        ct_list = []
        for i, b64 in enumerate(self.ciphertexts):
            path = f"/tmp/he_score_{i}.seal"
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            ct = ct_class()
            ct.load(context, path)
            os.remove(path)
            ct_list.append(ct)
        return ct_list

    @classmethod
    def from_dict(cls, d):
        return cls(d["ciphertexts"], d["n_rows"])


# ─────────────────────────────────────────────
# MESSAGE PARSER
# Single entry point to parse any incoming message
# ─────────────────────────────────────────────
class MessageParser:
    @staticmethod
    def parse(json_str):
        """
        Parse a JSON string into the correct Message subclass.
        Usage:
            msg = MessageParser.parse(json_str)
            if msg.msg_type == MessageType.INVERT_REQUEST:
                ...
        """
        d = json.loads(json_str)
        t = d["type"]

        if t == MessageType.DATA:
            return DataMessage.from_dict(d)
        elif t == MessageType.INVERT_REQUEST:
            return InvertRequestMessage.from_dict(d)
        elif t == MessageType.INVERT_RESPONSE:
            return InvertResponseMessage.from_dict(d)
        elif t == MessageType.SCORE:
            return ScoreMessage.from_dict(d)
        else:
            raise ValueError(f"Unknown message type: {t}")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("Message Format Test")
    print("=" * 50)

    # Test 1: DataMessage
    msg1 = DataMessage("TRAIN", ["ct_bytes_1", "ct_bytes_2"])
    j1   = msg1.to_json()
    msg1_back = MessageParser.parse(j1)
    print(f"\n[1] DataMessage")
    print(f"    type      : {msg1_back.msg_type}")
    print(f"    data_type : {msg1_back.data_type}")
    print(f"    n_cts     : {len(msg1_back.ciphertexts)}")

    # Test 2: InvertRequestMessage
    msg2 = InvertRequestMessage(InvertContext.N, ["ct_N_bytes"])
    j2   = msg2.to_json()
    msg2_back = MessageParser.parse(j2)
    print(f"\n[2] InvertRequestMessage")
    print(f"    type    : {msg2_back.msg_type}")
    print(f"    context : {msg2_back.context}")
    print(f"    n_cts   : {len(msg2_back.ciphertexts)}")

    # Test 3: InvertResponseMessage
    msg3 = InvertResponseMessage(InvertContext.N, ["ct_inv_N_bytes"])
    j3   = msg3.to_json()
    msg3_back = MessageParser.parse(j3)
    print(f"\n[3] InvertResponseMessage")
    print(f"    type    : {msg3_back.msg_type}")
    print(f"    context : {msg3_back.context}")
    print(f"    n_cts   : {len(msg3_back.ciphertexts)}")

    # Test 4: ScoreMessage
    msg4 = ScoreMessage(["score_ct_1", "score_ct_2"], n_rows=2)
    j4   = msg4.to_json()
    msg4_back = MessageParser.parse(j4)
    print(f"\n[4] ScoreMessage")
    print(f"    type   : {msg4_back.msg_type}")
    print(f"    n_rows : {msg4_back.n_rows}")
    print(f"    n_cts  : {len(msg4_back.ciphertexts)}")
"""
    print(f"\n  All message types working correctly")
    print(f"\n  Protocol flow:")
    print(f"    Client → Cloud  : DataMessage       (encrypted sensor data)")
    print(f"    Cloud  → Client : InvertRequest     (please invert this)")
    print(f"    Client → Cloud  : InvertResponse    (here is 1/value)")
    print(f"    Cloud  → Client : ScoreMessage      (here is anomaly score)")

"""