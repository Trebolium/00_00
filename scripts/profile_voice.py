"""Calibrate a per-user RMS energy threshold for VAD gating. Run: uv run python scripts/profile_voice.py"""
import json
import os
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

# Deliberately standalone -- no import from voice_agent.config/.audio_io, so this
# script needs no API keys. Must match SAMPLE_RATE/FRAME_MS in config.py.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
VOICE_PROFILE_PATH = os.getenv("VOICE_PROFILE_PATH", "voice_profile.json")


def frame_rms(frame: bytes) -> float:
    """RMS energy of a PCM16 frame -- kept in sync with voice_agent.audio_io.frame_rms."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def record_seconds(seconds: float) -> np.ndarray:
    """Blocking capture of `seconds` of mono int16 audio via sounddevice."""
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten()


def frame_rms_series(audio: np.ndarray) -> np.ndarray:
    """RMS energy per 20ms frame, dropping any incomplete trailing frame."""
    n_frames = len(audio) // FRAME_SAMPLES
    trimmed = audio[: n_frames * FRAME_SAMPLES]
    return np.array([frame_rms(trimmed[i : i + FRAME_SAMPLES].tobytes()) for i in range(0, len(trimmed), FRAME_SAMPLES)])


def derive_threshold(noise_rms: np.ndarray, speech_rms: np.ndarray):
    """Derive an energy threshold, or None if the calibration looks contaminated.

    Uses a 90th-percentile noise ceiling rather than mean + std, since a single loud
    outlier during the "stay quiet" step (a cough, a chair creak) badly inflates a
    mean/std-based estimate but barely moves a percentile.
    """
    noise_ceiling = float(np.percentile(noise_rms, 90))
    speech_median = float(np.median(speech_rms))
    if noise_ceiling >= speech_median:
        return None
    return {
        "energy_threshold": (noise_ceiling + speech_median) / 2,
        "noise_ceiling": noise_ceiling,
        "speech_median": speech_median,
    }


def main():
    print("Calibrating your voice profile for energy-gated VAD.")
    print("Step 1/2: stay quiet for 3 seconds (measuring background noise)...")
    time.sleep(1)
    noise = record_seconds(3)
    print("Got it.")

    print("Step 2/2: speak naturally for 6 seconds -- read a sentence, count numbers, etc.")
    time.sleep(1)
    speech = record_seconds(6)
    print("Got it.")

    result = derive_threshold(frame_rms_series(noise), frame_rms_series(speech))
    if result is None:
        print(
            "Calibration looks off: background noise was as loud as or louder than your "
            "typical speech (something loud probably happened during the 'stay quiet' step). "
            "Not saving -- please try again somewhere quieter."
        )
        return

    profile = {**result, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out_path = Path(VOICE_PROFILE_PATH)
    out_path.write_text(json.dumps(profile, indent=2))
    print(f"Saved threshold={result['energy_threshold']:.1f} to {out_path}")


if __name__ == "__main__":
    main()
