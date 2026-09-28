import json
import time

from .config import TRANSCRIPT_PATH


def append(role: str, text: str):
    """Append one turn to the transcript file as a JSON line."""
    with open(TRANSCRIPT_PATH, "a") as f:
        f.write(json.dumps({"role": role, "text": text, "ts": time.time()}) + "\n")
