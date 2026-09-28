from unittest.mock import MagicMock

import pytest

from voice_agent import tts


def test_synthesize_posts_to_elevenlabs_and_returns_pcm(monkeypatch):
    fake_response = MagicMock()
    fake_response.content = b"\x01\x02\x03\x04"
    fake_post = MagicMock(return_value=fake_response)
    monkeypatch.setattr(tts.requests, "post", fake_post)

    pcm, sample_rate = tts.synthesize("hello")

    assert pcm == b"\x01\x02\x03\x04"
    assert sample_rate == tts.SAMPLE_RATE
    fake_response.raise_for_status.assert_called_once()

    _, kwargs = fake_post.call_args
    assert kwargs["headers"]["xi-api-key"] == tts.ELEVEN_API_KEY
    assert kwargs["params"]["output_format"] == f"pcm_{tts.SAMPLE_RATE}"
    assert kwargs["json"]["text"] == "hello"


def test_synthesize_raises_on_http_error(monkeypatch):
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = tts.requests.HTTPError("boom")
    monkeypatch.setattr(tts.requests, "post", MagicMock(return_value=fake_response))

    with pytest.raises(tts.requests.HTTPError):
        tts.synthesize("hello")
