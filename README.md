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
  audio_io.py     # Mic (capture + VAD + energy gating + barge-in) and Speaker (playback)
  echo_cancel.py  # EchoCanceller: adaptive filter that removes mic-picked-up TTS echo
  asr.py          # transcribe(pcm_bytes) -> str
  llm.py          # respond(history) -> str
  tts.py          # synthesize(text) -> (pcm16_bytes, sample_rate)
  transcript.py   # append(role, text)
scripts/
  run.sh              # pulls secrets from Keychain, then runs the app
  profile_voice.py    # calibrates a per-user RMS energy threshold -> voice_profile.json
pyproject.toml    # project + dependencies (uv/hatchling)
Dockerfile
```

## Module map (swap points)

Each pipeline stage is one file with a single function/class, so any stage can be swapped independently:

- `audio_io.py` — `Mic` and `Speaker`. Swap for a different audio backend. `Mic` gates speech on webrtcvad's spectral check AND (if calibrated) an RMS energy threshold, to reject loud-but-non-speech noise/echo that VAD alone would pass.
- `echo_cancel.py` — `EchoCanceller`, a block-NLMS adaptive filter. Swap for a different echo-cancellation approach (e.g. a mature AEC library) if the built-in one isn't good enough.
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

### Voice calibration (optional)

webrtcvad's spectral check alone can be fooled by background noise or echo. Run a one-time calibration to add a second, energy-based gate tuned to how loud *your* voice actually is:

```bash
uv run python scripts/profile_voice.py
```

It records a few seconds of silence (noise floor) and a few seconds of you speaking, derives an RMS threshold between the two, and saves it to `voice_profile.json` (gitignored, per-user). `Mic` loads this file at startup automatically; a frame only counts as voiced if it passes both VAD and the energy threshold. If the file doesn't exist, `Mic` falls back to VAD-only gating exactly as before -- calibration is optional.

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
- VAD endpointing (`TRAILING_SILENCE_MS` in `config.py`) is a fixed silence timeout, not adaptive. `record_utterance` also has a hard ceiling (`MAX_UTTERANCE_MS`) so sustained noise/echo that never reads as silence can't hang the app forever waiting for a trailing pause that never comes.
- Single-user, single-process, local mic/speaker only.
- Echo cancellation (`echo_cancel.py`) is a lightweight, hand-rolled block-NLMS adaptive filter, not a production AEC: no explicit delay estimation (it relies on the filter length, `ECHO_FILTER_MS`, being long enough to span the actual speaker-to-mic delay), no double-talk detection, and TTS audio is resampled to the mic's rate with plain linear interpolation (no anti-aliasing). It meaningfully reduces false barge-ins caused by the mic hearing its own TTS output, combined with `BARGE_IN_MS` requiring sustained voiced audio rather than a single VAD frame — but it won't be perfect, and Whisper is known to hallucinate short stock phrases (e.g. "Thank you.") on any echo residue that does get through. Headphones sidestep the problem entirely. The filter includes basic stability safeguards (weight leakage, clipping, and a NaN/Inf self-heal) since a hand-rolled adaptive filter running on real speech can occasionally diverge — it resets to a safe state rather than permanently corrupting the mic pipeline.
