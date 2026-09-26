# voice-agent

A minimal voice-conversation loop: mic -> Groq Whisper (ASR) -> OpenRouter LLM -> LiveKit ElevenLabs TTS -> speaker, with barge-in support and a running transcript.

## How it works

1. `main.py` idles until the mic's VAD detects speech, then records the utterance until trailing silence.
2. That audio goes through the pipeline in `run_cycle()`: `asr.transcribe` -> `llm.respond` -> `tts.synthesize` -> `Speaker.play`.
3. While a turn is in flight (thinking or speaking), a second task (`Mic.wait_for_barge_in`) listens for new speech. Whichever finishes first wins the race in `run_cycle`; if the user starts talking, the in-flight turn is cancelled, playback is stopped immediately, and the loop starts recording the new utterance.
4. Every user/assistant turn is appended to `transcript.jsonl` as it happens.

## Module map (swap points)

Each pipeline stage is one file with a single function/class, so any stage can be swapped independently:

- `audio_io.py` — `Mic` (capture + VAD + barge-in) and `Speaker` (playback). Swap for a different audio backend.
- `asr.py` — `transcribe(pcm_bytes) -> str`. Currently Groq's free Whisper endpoint.
- `llm.py` — `respond(history) -> str`. Currently OpenRouter (`google/gemini-3-pro-preview` by default), via the OpenAI-compatible client. Change `LLM_MODEL` in `.env` to use a different model, or point `base_url` elsewhere entirely.
- `tts.py` — `synthesize(text) -> (pcm16_bytes, sample_rate)`. Currently the LiveKit Agents ElevenLabs plugin (`livekit.plugins.elevenlabs`), used standalone (no LiveKit room/server needed).
- `transcript.py` — `append(role, text)`. Currently a local JSONL file.

## Setup

Requires **Python 3.10+** (`livekit-agents` uses `typing.TypeAlias`). Tested on 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, OPENROUTER_API_KEY, ELEVEN_API_KEY
python main.py
```

## Known simplifications

- Barge-in cancels the in-flight ASR/LLM task, but a blocking network call already running in a worker thread can't be forcibly killed — it finishes in the background and its result is just discarded. Playback itself *is* stopped immediately (`Speaker.stop()`).
- VAD endpointing (`TRAILING_SILENCE_MS` in `config.py`) is a fixed silence timeout, not adaptive.
- Single-user, single-process, local mic/speaker only — no networking/rooms involved despite using a LiveKit plugin for TTS.
