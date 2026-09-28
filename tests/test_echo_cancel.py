import numpy as np

from voice_agent import config
from voice_agent.echo_cancel import FILTER_TAPS, EchoCanceller


def test_process_with_no_reference_leaves_signal_unchanged():
    canceller = EchoCanceller()
    mic_frame = np.array([100, -200, 300, -50], dtype=np.int16)

    residual = canceller.process(mic_frame)

    assert np.array_equal(residual, mic_frame)  # zero weights -> predicted echo is 0


def test_converges_to_cancel_a_pure_undelayed_echo():
    canceller = EchoCanceller()
    rng = np.random.default_rng(0)
    reference = (rng.uniform(-1, 1, size=6000) * 5000).astype(np.int16)

    frame_size = 320
    errors = []
    for i in range(0, len(reference) - frame_size, frame_size):
        chunk = reference[i : i + frame_size]
        canceller.push_reference(chunk, sample_rate=config.SAMPLE_RATE)
        residual = canceller.process(chunk)  # mic hears exactly the reference: a pure, undelayed echo
        errors.append(np.abs(residual.astype(np.float64)).mean())

    first_few = np.mean(errors[:3])
    last_few = np.mean(errors[-3:])
    assert last_few < first_few * 0.5  # the filter should have adapted to meaningfully cancel the echo


def test_push_reference_resamples_to_mic_rate():
    canceller = EchoCanceller()
    tts_chunk = np.full(2400, 1000, dtype=np.int16)  # e.g. 100ms at 24kHz

    canceller.push_reference(tts_chunk, sample_rate=24000)

    # resampled to the mic's 16kHz should be ~100ms worth of samples, not the original 2400
    expected_len = int(len(tts_chunk) * config.SAMPLE_RATE / 24000)
    assert abs(len(canceller._ref_history) - (canceller._weights.shape[0] + expected_len)) <= 1


def test_recovers_from_a_prior_divergence_instead_of_propagating_nan():
    canceller = EchoCanceller()
    canceller._weights[:] = np.inf  # simulate a prior instability, e.g. a numerical blow-up
    mic_frame = np.full(320, 1000, dtype=np.int16)
    canceller.push_reference(mic_frame, sample_rate=config.SAMPLE_RATE)

    residual = canceller.process(mic_frame)

    assert np.isfinite(residual).all()
    assert np.array_equal(residual, mic_frame)  # falls back to passthrough for this frame
    assert np.all(canceller._weights == 0)  # and resets to a safe state

    # a normal frame right after should work fine, proving the reset was real
    residual2 = canceller.process(mic_frame)
    assert np.isfinite(residual2).all()


def test_weights_stay_within_the_clip_bound_under_sustained_loud_input():
    canceller = EchoCanceller()
    loud = np.full(320, 32767, dtype=np.int16)

    for _ in range(100):
        canceller.push_reference(loud, sample_rate=config.SAMPLE_RATE)
        canceller.process(loud)

    assert np.isfinite(canceller._weights).all()
    assert np.max(np.abs(canceller._weights)) <= 10.0 + 1e-9


def test_leakage_slowly_decays_weights_with_no_reference_or_error():
    canceller = EchoCanceller()
    canceller._weights[:] = 1.0  # pretend the filter already converged to something

    silence = np.zeros(320, dtype=np.int16)
    canceller.push_reference(silence, sample_rate=config.SAMPLE_RATE)
    canceller.process(silence)  # error will be ~0 (predicted echo of silence is 0), so only leakage acts

    assert np.all(canceller._weights < 1.0)  # leakage nudged weights toward zero
    assert canceller._weights.shape == (FILTER_TAPS,)
