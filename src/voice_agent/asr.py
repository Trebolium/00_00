import io
import wave

from groq import Groq

from .config import GROQ_API_KEY, SAMPLE_RATE

_client = Groq(api_key=GROQ_API_KEY)


def transcribe(pcm_bytes: bytes) -> str:
    """Send raw PCM16 audio to Groq's Whisper endpoint and return the transcript text."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    result = _client.audio.transcriptions.create(
        file=("utterance.wav", wav_buf.getvalue()),
        model="whisper-large-v3-turbo",
    )
    return result.text.strip()
