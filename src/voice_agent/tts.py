import requests

from .config import ELEVEN_API_KEY, TTS_VOICE_ID

SAMPLE_RATE = 24000


def synthesize(text: str) -> tuple[bytes, int]:
    """Synthesize speech via ElevenLabs' REST API. Returns (pcm16_bytes, sample_rate)."""
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}",
        headers={"xi-api-key": ELEVEN_API_KEY},
        params={"output_format": f"pcm_{SAMPLE_RATE}"},
        json={"text": text, "model_id": "eleven_turbo_v2_5"},
    )
    response.raise_for_status()
    return response.content, SAMPLE_RATE
