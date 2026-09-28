# voice-agent

A minimal voice-conversation loop: mic -> Groq Whisper (ASR) -> OpenRouter LLM -> ElevenLabs TTS -> speaker, with barge-in support and a running transcript.

## How it works

1. `main.py` idles until the mic's VAD detects speech, then records the utterance until trailing silence.
2. That audio goes through the pipeline in `run_cycle()`: `asr.transcribe` -> `llm.respond` -> `tts.synthesize` -> `Speaker.play`.
3. While a turn is in flight (thinking or speaking), a second task (`Mic.wait_for_barge_in`) listens for new speech. Whichever finishes first wins the race in `run_cycle`; if the user starts talking, the in-flight turn is cancelled, playback is stopped immediately, and the loop starts recording the new utterance.
4. Every user/assistant turn is appended to `transcript.jsonl` as it happens.

## Layout

```
src/voice_agent/
  main.py        # orchestrator loop + entry point (run())
  audio_io.py     # Mic (capture + VAD + barge-in) and Speaker (playback)
  asr.py          # transcribe(pcm_bytes) -> str
  llm.py          # respond(history) -> str
  tts.py          # synthesize(text) -> (pcm16_bytes, sample_rate)
  transcript.py   # append(role, text)
pyproject.toml    # project + dependencies (uv/hatchling)
Dockerfile
```

## Module map (swap points)

Each pipeline stage is one file with a single function/class, so any stage can be swapped independently:

- `audio_io.py` — `Mic` and `Speaker`. Swap for a different audio backend.
- `asr.py` — Currently Groq's free Whisper endpoint.
- `llm.py` — Currently OpenRouter (`google/gemini-3-pro-preview` by default), via the OpenAI-compatible client. Change `LLM_MODEL` in `.env` to use a different model, or point `base_url` elsewhere entirely.
- `tts.py` — Currently a direct call to ElevenLabs' REST API (no SDK).
- `transcript.py` — Currently a local JSONL file.

## Setup (local, with uv)

Requires **Python 3.10+**; this repo is pinned to 3.12 via `.python-version`.

```bash
uv sync                    # creates .venv and installs everything from uv.lock
cp .env.example .env       # fill in GROQ_API_KEY, OPENROUTER_API_KEY, ELEVEN_API_KEY
uv run voice-agent
```

### Secrets via macOS Keychain (alternative to `.env`)

Instead of a plaintext `.env` file, you can store the three keys in Keychain and pull them into the process at run time:

```bash
security add-generic-password -a "$USER" -s groq-api-key -w "<value>"
security add-generic-password -a "$USER" -s openrouter-api-key -w "<value>"
security add-generic-password -a "$USER" -s eleven-api-key -w "<value>"

./scripts/run.sh   # looks the keys up and execs `uv run voice-agent`
```

## Testing

```bash
uv run pytest
```

Every external boundary (Groq, OpenRouter, ElevenLabs, PortAudio) is mocked, so the suite needs no real credentials, network access, or audio hardware — the dummy keys in `tests/conftest.py` just let the modules import.

## Docker

```bash
docker build -t voice-agent .
docker run --rm --env-file .env voice-agent
```

Note: this app talks to a real microphone and speaker (`sounddevice`/PortAudio on the host's audio devices). A headless cloud container has no audio hardware to attach, so the Docker image is set up for the packaging/deploy story (locked deps, small base image, non-interactive entry point) rather than for actually running the mic loop in the cloud as-is. Running it in a real cloud deployment would mean swapping `audio_io.py` for a networked audio source (e.g. audio streamed in over a WebSocket/WebRTC connection from a browser or phone client) — that's the intended swap point.

## Known simplifications

- Barge-in cancels the in-flight ASR/LLM task, but a blocking network call already running in a worker thread can't be forcibly killed — it finishes in the background and its result is just discarded. Playback itself *is* stopped immediately (`Speaker.stop()`).
- VAD endpointing (`TRAILING_SILENCE_MS` in `config.py`) is a fixed silence timeout, not adaptive.
- Single-user, single-process, local mic/speaker only.
