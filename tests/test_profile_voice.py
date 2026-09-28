import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "profile_voice.py"
_spec = importlib.util.spec_from_file_location("profile_voice", SCRIPT_PATH)
profile_voice = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_voice)


def test_derive_threshold_sits_between_noise_and_speech():
    rng = np.random.default_rng(0)
    noise_rms = 10.0 + rng.uniform(0, 5, size=150)  # a little natural variation, low baseline
    speech_rms = np.full(150, 200.0)

    result = profile_voice.derive_threshold(noise_rms, speech_rms)

    assert result is not None
    assert result["noise_ceiling"] < result["energy_threshold"] < result["speech_median"]


def test_derive_threshold_tolerates_a_single_brief_outlier():
    # one loud frame out of 150 (a single cough syllable) shouldn't move a percentile much
    noise_rms = np.array([10.0] * 149 + [5000.0])
    speech_rms = np.full(150, 200.0)

    result = profile_voice.derive_threshold(noise_rms, speech_rms)

    assert result is not None
    assert result["noise_ceiling"] < 50  # nowhere near dragged up to the outlier's level


def test_derive_threshold_rejects_sustained_loud_contamination():
    # a genuinely noisy "quiet" phase -- e.g. a cough/throat-clear lasting several
    # hundred ms, not just one brief blip -- should be caught, not silently accepted
    noise_rms = np.array([10.0] * 120 + [500.0] * 30)  # 20% of the recording is loud
    speech_rms = np.full(150, 200.0)

    result = profile_voice.derive_threshold(noise_rms, speech_rms)

    assert result is None
