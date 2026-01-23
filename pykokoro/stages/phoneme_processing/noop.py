from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


@dataclass
class NoopPhonemeProcessorAdapter:
    def process(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = (cfg, trace)
        return phoneme_segments
