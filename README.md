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
- `echo_cancel.py` — `EchoCanceller`, a block-NLMS adaptive filter. **Not currently wired up in `main.py`** (see "Known simplifications" below) -- kept as a starting point if a different echo-cancellation approach (e.g. a mature AEC library) is worth revisiting.
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

The noise-floor estimate uses a 90th-percentile ceiling (not mean + std), so a single loud outlier during the "stay quiet" step (a cough, a chair creak) doesn't badly skew it. If the background recording still comes out as loud as or louder than your actual speech, the script refuses to save a profile and asks you to try again somewhere quieter, rather than silently writing a threshold that would reject your real voice.

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
- VAD endpointing (`TRAILING_SILENCE_MS` in `config.py`) is a fixed silence timeout, not adaptive. `record_utterance` also has a hard ceiling (`MAX_UTTERANCE_MS`) so sustained noise/echo that never reads as silence can't hang the app forever waiting for a trailing pause that never comes, and a floor (`MIN_UTTERANCE_MS`, on total *voiced* content, not recording length) so a brief cough or throat-clear doesn't get sent to ASR and risk a Whisper hallucination.
- Single-user, single-process, local mic/speaker only.
- **Echo cancellation (`echo_cancel.py`) is disabled by default, not wired up in `main.py`.** It's a lightweight, hand-rolled block-NLMS adaptive filter -- in live testing it proved unstable in exactly the way it was meant to prevent: it repeatedly diverged and saturated its own output (individual filter weights stayed within their clip bound, but combined across `FILTER_TAPS` taps into a wildly saturating prediction that no single-value NaN/Inf check catches), producing large fabricated "voice" energy out of an actual mic signal that was measured as near-silent. It has stability safeguards (weight leakage, clipping, a self-heal that now also catches large-but-finite saturation, not just NaN/Inf) and passing tests for those safeguards in isolation, but that wasn't enough to make it reliable against real ElevenLabs TTS audio end-to-end. Given real measured mic levels during TTS playback were already low enough not to need cancellation in testing, `Mic`/`Speaker` are constructed without an `EchoCanceller` in `main.py`. It's left in the codebase as a starting point, not something to trust as-is -- revisit with a mature AEC library (e.g. WebRTC's AEC3, Speex) rather than iterating further on the hand-rolled version if echo turns out to be a real problem in some other setup.
- Barge-in false positives are now guarded by three independent, much simpler layers instead: `BARGE_IN_MS` (300ms of sustained voiced audio, not a single VAD frame), an optional per-user RMS energy threshold (`scripts/profile_voice.py`), and `MIN_UTTERANCE_MS` (rejects brief blips like coughs before they reach ASR). Whisper is still known to hallucinate short stock phrases (e.g. "Thank you.") on faint/ambiguous audio, so headphones remain the most reliable way to avoid mic-picked-up echo entirely.
