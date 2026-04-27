"""
run_network.py
==============
Entry point for HE-SAD protocol over network.

Run server first in Terminal 1:
    uvicorn server.http_server:app --host 0.0.0.0 --port 8000

Then run client in Terminal 2:
    python run_network.py
"""

from client.http_client import HEClient

if __name__ == "__main__":
    client = HEClient()
    client.run_sad()