from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import PhonemeSegment, Segment, Trace
from ..protocols import DocumentResult

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
                id=f"{segment.id}_ph0",
                segment_id=segment.id,
                phoneme_id=0,
                text=segment.text,
                phonemes=segment.text,
                tokens=[],
                lang=lang,
                char_start=segment.char_start,
                char_end=segment.char_end,
                paragraph_idx=segment.paragraph_idx,
                sentence_idx=segment.sentence_idx,
                clause_idx=segment.clause_idx,
                pause_before=0.0,
                pause_after=0.0,
            )
            for segment in segments
        ]
