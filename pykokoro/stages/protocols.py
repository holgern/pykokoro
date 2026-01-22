from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..types import AnnotationSpan, BoundaryEvent, PhonemeSegment, Segment, Trace

if TYPE_CHECKING:
    from ..pipeline_config import PipelineConfig


@dataclass
class DocumentResult:
    clean_text: str
    annotation_spans: list[AnnotationSpan] = field(default_factory=list)
    boundary_events: list[BoundaryEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DocumentParser(Protocol):
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult: ...


class Splitter(Protocol):
    def split(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]: ...


class G2PAdapter(Protocol):
    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]: ...


class Synthesizer(Protocol):
    def synthesize(
        self, phoneme_segments: list[PhonemeSegment], cfg: PipelineConfig, trace: Trace
    ) -> np.ndarray: ...
