from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...constants import SAMPLE_RATE
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


@dataclass
class NoopAudioGenerationAdapter:
    seconds_per_segment: float = 0.1

    def generate(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = (cfg, trace)
        samples = int(SAMPLE_RATE * self.seconds_per_segment)
        for segment in phoneme_segments:
            if not segment.phonemes.strip():
                segment.raw_audio = None
                segment.processed_audio = None
                continue
            segment.raw_audio = np.zeros(samples, dtype=np.float32)
            segment.processed_audio = None
        return phoneme_segments
