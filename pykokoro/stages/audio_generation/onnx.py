from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx_backend import Kokoro
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    import numpy as np

    from ...pipeline_config import PipelineConfig


class OnnxAudioGenerationAdapter:
    def __init__(self, kokoro: Kokoro) -> None:
        self._kokoro = kokoro

    def generate(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = trace
        voice_style = self._kokoro.resolve_voice_style(cfg.voice)

        def voice_resolver(voice_name: str) -> np.ndarray:
            return self._kokoro.get_voice_style(voice_name)

        return self._kokoro.generate_raw_audio_segments(
            phoneme_segments,
            voice_style,
            cfg.generation.speed,
            voice_resolver,
        )
