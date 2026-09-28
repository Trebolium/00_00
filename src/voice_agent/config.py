import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
ELEVEN_API_KEY = os.environ["ELEVEN_API_KEY"]

LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # ElevenLabs "Rachel"

SAMPLE_RATE = 16000  # mic capture rate, required by webrtcvad
FRAME_MS = 20  # webrtcvad only accepts 10/20/30ms frames
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive about filtering non-speech
TRAILING_SILENCE_MS = 700  # silence needed to end an utterance

TRANSCRIPT_PATH = os.getenv("TRANSCRIPT_PATH", "transcript.jsonl")
