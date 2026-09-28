from unittest.mock import MagicMock

from voice_agent import asr


def test_transcribe_wraps_pcm_as_wav_and_returns_stripped_text(monkeypatch):
    fake_response = MagicMock()
    fake_response.text = "  hello world  \n"
    fake_create = MagicMock(return_value=fake_response)
    monkeypatch.setattr(asr._client.audio.transcriptions, "create", fake_create)

    silence = b"\x00\x00" * 320  # one 20ms frame of int16 silence
    result = asr.transcribe(silence)

    assert result == "hello world"
    fake_create.assert_called_once()
    _, kwargs = fake_create.call_args
    assert kwargs["model"] == "whisper-large-v3-turbo"
    filename, wav_bytes = kwargs["file"]
    assert filename == "utterance.wav"
    assert wav_bytes.startswith(b"RIFF")  # valid WAV container, not raw PCM
