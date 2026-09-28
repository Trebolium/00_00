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
MAX_UTTERANCE_MS = 15000  # hard cap so sustained noise/echo can't make record_utterance hang forever
MIN_UTTERANCE_MS = 300  # minimum total voiced content for an utterance to be sent to ASR, not a cough/blip
BARGE_IN_MS = 300  # sustained voiced audio needed to count as a real interruption, not mic-picked-up echo

ECHO_FILTER_MS = 150  # how much echo-path delay/reverb the NLMS canceller can model
ECHO_STEP_SIZE = 0.15  # NLMS adaptation rate (0-1]; higher = faster but less stable

TRANSCRIPT_PATH = os.getenv("TRANSCRIPT_PATH", "transcript.jsonl")
VOICE_PROFILE_PATH = os.getenv("VOICE_PROFILE_PATH", "voice_profile.json")  # per-user RMS calibration, see scripts/profile_voice.py
