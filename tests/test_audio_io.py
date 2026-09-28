import threading
import time

import numpy as np
import pytest

from voice_agent import audio_io


class _FakeInputStream:
    """Stands in for sounddevice.InputStream so tests don't need real audio hardware."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    monkeypatch.setattr(audio_io.sd, "InputStream", _FakeInputStream)


def make_mic():
    return audio_io.Mic()


def test_wait_for_speech_skips_silence_and_returns_first_voiced_frame():
    mic = make_mic()

    # wait_for_speech() drains any stale queued frames on entry, so frames must
    # arrive *after* it starts blocking on get() -- not be pre-queued.
    def push_frames_once_blocked():
        time.sleep(0.05)
        mic._frames.put((b"silence-1", False))
        mic._frames.put((b"silence-2", False))
        mic._frames.put((b"voice-1", True))

    threading.Thread(target=push_frames_once_blocked, daemon=True).start()

    assert mic.wait_for_speech() == b"voice-1"


def test_record_utterance_stops_after_trailing_silence():
    mic = make_mic()
    frame_ms = audio_io.FRAME_MS
    silence_frames_needed = audio_io.TRAILING_SILENCE_MS // frame_ms

    # one more voiced frame, then exactly enough silence to close the utterance
    mic._frames.put((b"voice-2", True))
    for i in range(silence_frames_needed):
        mic._frames.put((f"silence-{i}".encode(), False))
    # extra frame after closing that must NOT be consumed
    mic._frames.put((b"should-not-be-read", True))

    pcm = mic.record_utterance(first_frame=b"voice-1")

    assert pcm.startswith(b"voice-1voice-2")
    assert mic._frames.qsize() == 1  # the trailing extra frame is left untouched


def test_record_utterance_resets_silence_run_on_interleaved_voice():
    mic = make_mic()
    frame_ms = audio_io.FRAME_MS
    silence_frames_needed = audio_io.TRAILING_SILENCE_MS // frame_ms

    # not-quite-enough silence, interrupted by voice, then a full silence run
    for i in range(silence_frames_needed - 1):
        mic._frames.put((f"s1-{i}".encode(), False))
    mic._frames.put((b"voice-again", True))
    for i in range(silence_frames_needed):
        mic._frames.put((f"s2-{i}".encode(), False))

    pcm = mic.record_utterance(first_frame=b"voice-1")

    assert b"voice-again" in pcm


def test_callback_sets_barge_in_event_only_when_armed():
    mic = make_mic()
    monkeypatch_vad_is_speech(mic, voiced=True)

    indata = np.zeros(audio_io.FRAME_SAMPLES, dtype=np.int16)
    mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert not mic._speech_started.is_set()  # not armed yet

    mic._barge_armed = True
    mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert mic._speech_started.is_set()


def test_wait_for_barge_in_unblocks_when_event_set():
    mic = make_mic()
    threading.Timer(0.05, mic._speech_started.set).start()

    mic.wait_for_barge_in()  # would hang forever if barge-in detection were broken

    assert mic._barge_armed is False  # disarmed again on the way out


def monkeypatch_vad_is_speech(mic, voiced: bool):
    mic._vad.is_speech = lambda frame, rate: voiced


def test_speaker_stop_halts_playback_before_all_chunks_are_written(monkeypatch):
    written_chunks = []

    class _FakeOutputStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def write(self, chunk):
            written_chunks.append(chunk)
            speaker.stop()  # simulate the user interrupting mid-playback

    monkeypatch.setattr(audio_io.sd, "OutputStream", lambda **kwargs: _FakeOutputStream())
    monkeypatch.setattr(audio_io.sd, "stop", lambda: None)

    speaker = audio_io.Speaker()
    pcm = (np.arange(2000, dtype=np.int16)).tobytes()

    speaker.play(pcm, sample_rate=16000)

    assert len(written_chunks) == 1  # stopped after the first chunk, not all of them


def test_speaker_play_is_noop_for_empty_audio(monkeypatch):
    monkeypatch.setattr(
        audio_io.sd,
        "OutputStream",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not open a stream")),
    )
    audio_io.Speaker().play(b"", sample_rate=16000)
