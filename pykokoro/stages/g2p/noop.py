from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import Segment, Trace
from ..protocols import DocumentResult
from .kokorog2p import PhonemeSegment

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


class NoopG2PAdapter:
    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = (doc, trace)
        lang = cfg.generation.lang
        return [
            PhonemeSegment(
                text=segment.text,
                phonemes=segment.text,
                tokens=[],
                lang=lang,
                paragraph=segment.paragraph_idx or 0,
                sentence=segment.sentence_idx,
                pause_before=0.0,
                pause_after=0.0,
            )
            for segment in segments
        ]
