from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class AnnotationSpan:
    """Span-based markup annotation (character offsets refer to clean_text)."""

    char_start: int
    char_end: int
    attrs: dict[str, str]


@dataclass(frozen=True)
class BoundaryEvent:
    """Boundary event for SSMD breaks or markers."""

    pos: int
    kind: Literal["pause", "marker"]
    duration_s: float | None = None
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    """A chunk of input text with stable offsets into the document."""

    id: str
    text: str
    char_start: int
    char_end: int
    meta: dict[str, Any] = field(default_factory=dict)
    paragraph_idx: int | None = None
    sentence_idx: int | None = None
    clause_idx: int | None = None


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    name: str
    ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Structured debugging output."""

    warnings: list[str] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)


@dataclass
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    segments: list[Segment] = field(default_factory=list)
    trace: Trace | None = None

    def save_wav(self, path: str) -> None:
        """Save 16-bit PCM WAV. Minimal dependency implementation."""
        import wave

        x = self.audio
        if x.dtype != np.int16:
            x = np.clip(x, -1.0, 1.0)
            x = (x * 32767.0).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(x.tobytes())


# Backward compatibility aliases
Annotation = AnnotationSpan
