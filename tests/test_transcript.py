"""Tests for transcript compile and validation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pykokoro import Kokoro, export_transcript, validate_transcript
from pykokoro.constants import SAMPLE_RATE


def test_compile_basic_transcript():
    kokoro = Kokoro()
    transcript = kokoro.compile("Hello world", voice="af_sarah")

    assert transcript["format_version"] == "1.0"
    assert transcript["defaults"]["lang"] == "en-us"
    assert transcript["defaults"]["voice"]["name"] == "af_sarah"
    assert transcript["segments"]
    assert transcript["segments"][0]["phonemes"]


def test_compile_say_as_metadata():
    kokoro = Kokoro()
    transcript = kokoro.compile("[123]{as='cardinal'}", voice="af_sarah")

    segment = transcript["segments"][0]
    assert segment["flags"]["say_as_applied"] is True
    assert segment["metadata"]["say_as"]["interpret_as"] == "cardinal"


def test_validate_transcript_version_error():
    kokoro = Kokoro()
    transcript = kokoro.compile("Hello", voice="af_sarah")
    transcript["format_version"] = "0.0"

    with pytest.raises(ValueError, match="format_version"):
        validate_transcript(transcript)


def test_create_from_transcript_json(monkeypatch):
    kokoro = Kokoro()
    transcript = kokoro.compile("Hello", voice="af_sarah")
    transcript_json = export_transcript(transcript)

    def mock_init(self):
        self._audio_generator = SimpleNamespace(
            generate_from_segments=lambda *args, **kwargs: np.array(
                [0.0], dtype=np.float32
            )
        )
        self._voices_data = {"af_sarah": np.zeros(10)}
        self._session = object()

    monkeypatch.setattr(Kokoro, "_init_kokoro", mock_init)
    monkeypatch.setattr(Kokoro, "get_voice_style", lambda self, name: np.zeros(10))

    audio, sample_rate = kokoro.create_from_transcript(transcript_json)
    assert sample_rate == SAMPLE_RATE
    assert audio.shape == (1,)
