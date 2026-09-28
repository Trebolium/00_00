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


@pytest.fixture(autouse=True)
def no_real_voice_profile(monkeypatch, tmp_path):
    # Mic() must not pick up a real voice_profile.json from disk (e.g. if someone
    # actually ran calibration in this repo). Only patches the no-arg default path --
    # tests calling load_energy_threshold(path) with an explicit path are unaffected.
    original = audio_io.load_energy_threshold
    missing_path = str(tmp_path / "no_profile_here.json")
    monkeypatch.setattr(audio_io, "load_energy_threshold", lambda path=missing_path: original(path))


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


def test_record_utterance_stops_after_trailing_silence(monkeypatch):
    monkeypatch.setattr(audio_io, "MIN_UTTERANCE_MS", 0)  # this test isn't about the min-duration gate
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


def test_record_utterance_resets_silence_run_on_interleaved_voice(monkeypatch):
    monkeypatch.setattr(audio_io, "MIN_UTTERANCE_MS", 0)  # this test isn't about the min-duration gate
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


def test_record_utterance_returns_none_for_a_brief_non_speech_blip():
    mic = make_mic()
    silence_needed = audio_io.TRAILING_SILENCE_MS // audio_io.FRAME_MS

    # only the single first_frame is voiced (e.g. a cough) -- everything after is silence
    for i in range(silence_needed):
        mic._frames.put((f"silence-{i}".encode(), False))

    pcm = mic.record_utterance(first_frame=b"cough")

    assert pcm is None


def test_record_utterance_returns_bytes_for_sustained_real_speech():
    mic = make_mic()
    silence_needed = audio_io.TRAILING_SILENCE_MS // audio_io.FRAME_MS
    min_voiced_frames = audio_io.MIN_UTTERANCE_MS // audio_io.FRAME_MS

    for i in range(min_voiced_frames):  # enough sustained voice to clear the minimum
        mic._frames.put((f"voice-{i}".encode(), True))
    for i in range(silence_needed):
        mic._frames.put((f"silence-{i}".encode(), False))

    pcm = mic.record_utterance(first_frame=b"voice-first")

    assert pcm is not None
    assert b"voice-first" in pcm


def test_record_utterance_stops_at_max_duration_even_without_silence(monkeypatch):
    monkeypatch.setattr(audio_io, "MAX_UTTERANCE_MS", 100)  # small cap so the test stays fast
    mic = make_mic()
    max_frames = audio_io.MAX_UTTERANCE_MS // audio_io.FRAME_MS

    # continuously voiced, never silent -- would hang forever without the cap
    for i in range(max_frames + 50):
        mic._frames.put((f"voice-{i}".encode(), True))

    mic.record_utterance(first_frame=b"voice-first")

    # consumed exactly max_frames-1 (first_frame already counts as 1), leaving the rest untouched
    assert mic._frames.qsize() == 51


def test_callback_ignores_brief_blips_but_fires_after_sustained_voice():
    mic = make_mic()
    monkeypatch_vad_is_speech(mic, voiced=True)
    frames_needed = audio_io.BARGE_IN_MS // audio_io.FRAME_MS
    indata = np.zeros(audio_io.FRAME_SAMPLES, dtype=np.int16)

    mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert not mic._speech_started.is_set()  # not armed yet

    mic._barge_armed = True
    for _ in range(frames_needed - 1):
        mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert not mic._speech_started.is_set()  # a brief blip (e.g. mic-picked-up echo) isn't enough

    mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert mic._speech_started.is_set()  # sustained voice does trigger it


def test_callback_resets_run_on_interleaved_silence():
    mic = make_mic()
    frames_needed = audio_io.BARGE_IN_MS // audio_io.FRAME_MS
    indata = np.zeros(audio_io.FRAME_SAMPLES, dtype=np.int16)
    mic._barge_armed = True

    monkeypatch_vad_is_speech(mic, voiced=True)
    for _ in range(frames_needed - 1):
        mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)

    monkeypatch_vad_is_speech(mic, voiced=False)
    mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)  # one silent frame resets the run

    monkeypatch_vad_is_speech(mic, voiced=True)
    for _ in range(frames_needed - 1):
        mic._callback(indata, audio_io.FRAME_SAMPLES, None, None)
    assert not mic._speech_started.is_set()  # still short of a full sustained run after the reset


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


def test_frame_rms_computes_root_mean_square():
    silent = np.zeros(10, dtype=np.int16).tobytes()
    assert audio_io.frame_rms(silent) == 0.0

    loud = (np.ones(4, dtype=np.int16) * 3).tobytes()  # rms of [3,3,3,3] == 3
    assert audio_io.frame_rms(loud) == pytest.approx(3.0)


def test_load_energy_threshold_missing_file_returns_none(tmp_path):
    missing = tmp_path / "no_such_profile.json"
    assert audio_io.load_energy_threshold(str(missing)) is None


def test_load_energy_threshold_reads_saved_value(tmp_path):
    profile = tmp_path / "voice_profile.json"
    profile.write_text('{"energy_threshold": 250.0}')
    assert audio_io.load_energy_threshold(str(profile)) == 250.0


def test_callback_falls_back_to_vad_only_when_no_energy_threshold():
    mic = make_mic()
    mic._energy_threshold = None
    monkeypatch_vad_is_speech(mic, voiced=True)

    quiet_frame = np.zeros(audio_io.FRAME_SAMPLES, dtype=np.int16)  # rms == 0, would fail any threshold
    mic._callback(quiet_frame, audio_io.FRAME_SAMPLES, None, None)

    _, voiced = mic._frames.get_nowait()
    assert voiced is True


def test_callback_rejects_quiet_frame_that_passes_vad():
    mic = make_mic()
    mic._energy_threshold = 1000.0
    monkeypatch_vad_is_speech(mic, voiced=True)  # VAD says speech, but frame is too quiet

    quiet_frame = np.zeros(audio_io.FRAME_SAMPLES, dtype=np.int16)
    mic._callback(quiet_frame, audio_io.FRAME_SAMPLES, None, None)

    _, voiced = mic._frames.get_nowait()
    assert voiced is False


def test_callback_accepts_frame_passing_both_vad_and_energy():
    mic = make_mic()
    mic._energy_threshold = 1000.0
    monkeypatch_vad_is_speech(mic, voiced=True)

    loud_frame = np.full(audio_io.FRAME_SAMPLES, 5000, dtype=np.int16)
    mic._callback(loud_frame, audio_io.FRAME_SAMPLES, None, None)

    _, voiced = mic._frames.get_nowait()
    assert voiced is True


def test_speaker_play_is_noop_for_empty_audio(monkeypatch):
    monkeypatch.setattr(
        audio_io.sd,
        "OutputStream",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not open a stream")),
    )
    audio_io.Speaker().play(b"", sample_rate=16000)
