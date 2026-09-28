import os

import pytest

# Dummy credentials so config.py can import without touching Keychain/.env/real
# services; every test mocks the network boundary instead of calling out for real.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("ELEVEN_API_KEY", "test-eleven-key")


@pytest.fixture(autouse=True)
def no_ambient_base_url_overrides(monkeypatch):
    """Guard against stray *_BASE_URL env vars (e.g. leftover shell exports) breaking tests."""
    for var in ("GROQ_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
