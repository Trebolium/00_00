import numpy as np
from livekit.plugins import elevenlabs

from .config import ELEVEN_API_KEY, TTS_VOICE_ID

_tts = elevenlabs.TTS(api_key=ELEVEN_API_KEY, voice_id=TTS_VOICE_ID)


async def synthesize(text: str) -> tuple[bytes, int]:
    """Synthesize speech for text via LiveKit's ElevenLabs TTS plugin. Returns (pcm16_bytes, sample_rate)."""
    chunks = []
    sample_rate = 22050
    async with _tts.synthesize(text) as stream:
        async for event in stream:
            sample_rate = event.frame.sample_rate
            chunks.append(np.frombuffer(event.frame.data, dtype=np.int16))
    pcm = np.concatenate(chunks).tobytes() if chunks else b""
    return pcm, sample_rate
