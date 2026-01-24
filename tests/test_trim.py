"""Tests for trim helper edge cases."""

import numpy as np
import pytest

from pykokoro.trim import ParameterError, energy_based_vad, frame_signal


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1)],
)
def test_energy_based_vad_short_inputs(length, expected_frames):
    audio = np.ones(length, dtype=np.float32)
    voice_activity = energy_based_vad(
        audio,
        sample_rate=1000,
        frame_duration_ms=10,
        energy_threshold=0.5,
    )
    assert voice_activity.shape == (expected_frames,)


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1)],
)
def test_frame_signal_short_inputs(length, expected_frames):
    audio = np.arange(length, dtype=np.float32)
    frames = frame_signal(
        audio,
        sample_rate=1000,
        frame_ms=10,
        hop_ms=5,
    )
    assert frames.shape == (expected_frames, 10)

    if length > 0:
        assert frames[0, 0] == audio[0]


def test_frame_signal_invalid_params():
    audio = np.ones(10, dtype=np.float32)
    with pytest.raises(ParameterError, match="frame_length"):
        frame_signal(audio, sample_rate=1000, frame_ms=0, hop_ms=10)
    with pytest.raises(ParameterError, match="hop_length"):
        frame_signal(audio, sample_rate=1000, frame_ms=10, hop_ms=0)
