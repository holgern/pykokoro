from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Annotation:
    """Span-based markup annotation (character offsets refer to clean_text)."""

    char_start: int
    char_end: int
    attrs: dict[str, str]


@dataclass(frozen=True)
class Segment:
    """A chunk of input text with stable offsets into the *original* document."""

    id: str
    text: str
    char_start: int
    char_end: int
    paragraph_idx: int | None = None
    sentence_idx: int | None = None
    clause_idx: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhonemeSegment:
    """A segment ready for synthesis."""

    segment_id: str
    # Exactly one of these should be provided.
    phonemes: str | None = None
    token_ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    stage: Literal["split", "markup", "g2p", "synth", "cache"]
    name: str
    ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Structured debugging output."""

    events: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Optional snapshots
    segments: list[Segment] | None = None
    clean_texts: list[str] | None = None
    annotations: list[list[Annotation]] | None = None
    phoneme_segments: list[PhonemeSegment] | None = None


@dataclass
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    segments: list[Segment] = field(default_factory=list)
    trace: Trace | None = None

    def save_wav(self, path: str) -> None:
        """Save 16-bit PCM WAV. Minimal dependency implementation."""
        import wave

        # Convert to int16 PCM
        x = self.audio
        if x.dtype != np.int16:
            x = np.clip(x, -1.0, 1.0)
            x = (x * 32767.0).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(x.tobytes())
