import threading

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import ECHO_FILTER_MS, ECHO_STEP_SIZE, SAMPLE_RATE

FILTER_TAPS = int(SAMPLE_RATE * ECHO_FILTER_MS / 1000)
_EPSILON = 1e-6  # avoids divide-by-zero when the reference is silent
_LEAKAGE = 0.9999  # slowly decays weights toward zero, bounding long-term drift
_WEIGHT_CLIP = 10.0  # hard safety bound; a real echo-path gain never needs to exceed this
_MAX_SANE_ERROR = 20000  # well above real audio's dynamic range; anything past this is a diverging filter,
# not real signal -- individually-clipped weights can still combine (dot product across
# FILTER_TAPS taps) into a large-but-finite prediction that saturates the output without
# any single value ever being NaN/Inf, so isfinite() alone doesn't catch this case.


def _resample(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Naive linear-interpolation resample -- fine for a reference signal, not for playback quality."""
    if orig_rate == target_rate or len(samples) == 0:
        return samples.astype(np.float64)
    target_n = int(len(samples) * target_rate / orig_rate)
    orig_idx = np.arange(len(samples))
    target_idx = np.linspace(0, len(samples) - 1, num=target_n)
    return np.interp(target_idx, orig_idx, samples.astype(np.float64))


class EchoCanceller:
    """Adaptive (block NLMS) filter that subtracts known TTS-playback echo from mic audio.

    Works because we already know the exact reference signal -- what we just sent to
    the speaker -- via push_reference(). This isn't a production-grade AEC (no delay
    estimation, no double-talk detection): it's just enough to stop the mic hearing its
    own TTS output as a false barge-in. Includes basic stability safeguards (weight
    leakage, clipping, self-heal on NaN/Inf or large-but-finite saturation) since a
    hand-rolled adaptive filter running on real speech can occasionally diverge.

    Not currently wired up in main.py -- it proved unstable enough in practice
    (diverged to saturating output even when raw mic input was near-silent) that it's
    disabled by default. See the README's "Known simplifications" section.
    """

    def __init__(self):
        self._weights = np.zeros(FILTER_TAPS)
        self._ref_history = np.zeros(FILTER_TAPS)
        self._lock = threading.Lock()

    def push_reference(self, samples: np.ndarray, sample_rate: int):
        """Feed newly-played TTS audio in (resampled to the mic's rate) so the canceller knows what's echoing."""
        resampled = _resample(samples, sample_rate, SAMPLE_RATE)
        with self._lock:
            self._ref_history = np.concatenate([self._ref_history, resampled])
            keep = FILTER_TAPS + len(resampled) + 4000  # bounded lookback so this never grows unbounded
            if len(self._ref_history) > keep:
                self._ref_history = self._ref_history[-keep:]

    def process(self, mic_frame: np.ndarray) -> np.ndarray:
        """Return mic_frame with predicted echo subtracted, adapting the filter as it goes."""
        n = len(mic_frame)
        with self._lock:
            history = self._ref_history
            if len(history) < FILTER_TAPS + n - 1:
                history = np.concatenate([np.zeros(FILTER_TAPS + n - 1 - len(history)), history])

            # row i = the FILTER_TAPS reference samples ending at mic sample i (a view, no copy)
            windows = sliding_window_view(history[-(FILTER_TAPS + n - 1) :], FILTER_TAPS)

            predicted = windows @ self._weights
            error = mic_frame.astype(np.float64) - predicted

            if not np.isfinite(error).all() or np.abs(error).max() > _MAX_SANE_ERROR:
                # weights from a prior unstable update have poisoned this frame's
                # prediction -- reset to a safe state instead of returning garbage forever.
                self._weights = np.zeros(FILTER_TAPS)
                return mic_frame

            # per-row NLMS normalization (each row by its own window's energy), summed
            # across the block -- normalizing by the block's total energy instead would
            # make updates ~n times too small.
            row_energy = np.sum(windows**2, axis=1) + _EPSILON
            update = ECHO_STEP_SIZE * (windows.T @ (error / row_energy))

            self._weights = self._weights * _LEAKAGE + update
            np.clip(self._weights, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=self._weights)
            if not np.isfinite(self._weights).all():
                self._weights = np.zeros(FILTER_TAPS)

        return np.clip(error, -32768, 32767).astype(np.int16)
