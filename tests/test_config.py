import importlib
import sys

import pytest


def test_missing_required_key_raises_on_import(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)
    monkeypatch.setattr(sys, "path", sys.path)  # keep import machinery untouched
    sys.modules.pop("voice_agent.config", None)

    with pytest.raises(KeyError):
        importlib.import_module("voice_agent.config")

    # restore for any tests that import voice_agent.config afterwards
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("ELEVEN_API_KEY", "test-eleven-key")
    sys.modules.pop("voice_agent.config", None)
    importlib.import_module("voice_agent.config")
