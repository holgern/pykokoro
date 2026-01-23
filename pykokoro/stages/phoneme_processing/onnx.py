from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx_backend import Kokoro
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


class OnnxPhonemeProcessorAdapter:
    def __init__(self, kokoro: Kokoro) -> None:
        self._kokoro = kokoro

    def process(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = trace
        self._kokoro._init_kokoro()
        assert self._kokoro._audio_generator is not None
        return self._kokoro._audio_generator._preprocess_segments(
            phoneme_segments,
            cfg.generation.enable_short_sentence,
        )
