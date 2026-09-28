import json
import queue
import threading

import numpy as np
import sounddevice as sd
import webrtcvad

from .config import (
    BARGE_IN_MS,
    FRAME_MS,
    MAX_UTTERANCE_MS,
    SAMPLE_RATE,
    TRAILING_SILENCE_MS,
    VAD_AGGRESSIVENESS,
    VOICE_PROFILE_PATH,
)

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def frame_rms(frame: bytes) -> float:
    """RMS energy of a PCM16 frame, matching the units scripts/profile_voice.py calibrates against."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def load_energy_threshold(path: str = VOICE_PROFILE_PATH):
    """Load the calibrated RMS threshold, or None if no profile exists yet (VAD-only fallback)."""
    try:
        with open(path) as f:
            return json.load(f)["energy_threshold"]
    except (FileNotFoundError, KeyError, ValueError):
        return None


class Mic:
    """Continuously listens to the input device and exposes VAD-gated speech events."""

    def __init__(self, echo_canceller=None):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._energy_threshold = load_energy_threshold()  # None => VAD-only gating
        self._echo_canceller = echo_canceller
        self._frames = queue.Queue()
        self._speech_started = threading.Event()
        self._barge_armed = False
        self._barge_run = 0
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        samples = indata.reshape(-1)
        if self._echo_canceller is not None:
            samples = self._echo_canceller.process(samples)
        frame = samples.tobytes()
        voiced = self._vad.is_speech(frame, SAMPLE_RATE)
        if voiced and self._energy_threshold is not None:
            voiced = frame_rms(frame) >= self._energy_threshold
        self._frames.put((frame, voiced))
        if self._barge_armed:
            self._barge_run = self._barge_run + 1 if voiced else 0
            if self._barge_run * FRAME_MS >= BARGE_IN_MS:
                self._speech_started.set()

    def _drain(self):
        while not self._frames.empty():
            self._frames.get_nowait()

    def wait_for_speech(self):
        """Block until voiced audio arrives; returns that first voiced frame."""
        self._drain()
        while True:
            frame, voiced = self._frames.get()
            if voiced:
                return frame

    def record_utterance(self, first_frame):
        """Record from first_frame until trailing silence closes the utterance, or a max duration is hit.

        The max-duration cap exists so sustained noise/echo that keeps reading as
        "voiced" (silence never arriving) can't make this loop hang forever.
        """
        silence_needed = TRAILING_SILENCE_MS // FRAME_MS
        max_frames = MAX_UTTERANCE_MS // FRAME_MS
        buf = [first_frame]
        silence_run = 0
        while silence_run < silence_needed and len(buf) < max_frames:
            frame, voiced = self._frames.get()
            buf.append(frame)
            silence_run = 0 if voiced else silence_run + 1
        return b"".join(buf)

    def wait_for_barge_in(self):
        """Block until sustained new speech starts, used to detect interruption during LLM/TTS."""
        self._drain()
        self._speech_started.clear()
        self._barge_run = 0
        self._barge_armed = True
        self._speech_started.wait()
        self._barge_armed = False


class Speaker:
    """Plays PCM16 audio in small chunks and can be halted mid-playback for barge-in."""

    def __init__(self, echo_canceller=None):
        self._echo_canceller = echo_canceller
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        sd.stop()

    def play(self, pcm_bytes: bytes, sample_rate: int):
        if not pcm_bytes:
            return
        self._stop.clear()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunk = max(sample_rate // 10, 1)  # ~100ms chunks so stop() is responsive
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
            for i in range(0, len(audio), chunk):
                if self._stop.is_set():
                    break
                piece = audio[i : i + chunk]
                if self._echo_canceller is not None:
                    self._echo_canceller.push_reference(piece, sample_rate)
                stream.write(piece)
