import queue
import threading

import numpy as np
import sounddevice as sd
import webrtcvad

from .config import FRAME_MS, SAMPLE_RATE, TRAILING_SILENCE_MS, VAD_AGGRESSIVENESS

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class Mic:
    """Continuously listens to the input device and exposes VAD-gated speech events."""

    def __init__(self):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._frames = queue.Queue()
        self._speech_started = threading.Event()
        self._barge_armed = False
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
        frame = indata.tobytes()
        voiced = self._vad.is_speech(frame, SAMPLE_RATE)
        self._frames.put((frame, voiced))
        if voiced and self._barge_armed:
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
        """Record from first_frame until trailing silence closes the utterance. Returns PCM16 bytes."""
        silence_needed = TRAILING_SILENCE_MS // FRAME_MS
        buf = [first_frame]
        silence_run = 0
        while silence_run < silence_needed:
            frame, voiced = self._frames.get()
            buf.append(frame)
            silence_run = 0 if voiced else silence_run + 1
        return b"".join(buf)

    def wait_for_barge_in(self):
        """Block until new speech starts, used to detect interruption during LLM/TTS."""
        self._drain()
        self._speech_started.clear()
        self._barge_armed = True
        self._speech_started.wait()
        self._barge_armed = False


class Speaker:
    """Plays PCM16 audio in small chunks and can be halted mid-playback for barge-in."""

    def __init__(self):
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
                stream.write(audio[i : i + chunk])
