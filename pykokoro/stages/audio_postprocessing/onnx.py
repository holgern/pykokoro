from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx_backend import Kokoro
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    import numpy as np

    from ...pipeline_config import PipelineConfig


class OnnxAudioPostprocessingAdapter:
    def __init__(self, kokoro: Kokoro) -> None:
        self._kokoro = kokoro

    def postprocess(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> np.ndarray:
        _ = trace
        self._kokoro._init_kokoro()
        assert self._kokoro._audio_generator is not None
        trim_silence = cfg.generation.pause_mode == "manual"
        processed = self._kokoro._audio_generator._postprocess_audio_segments(
            phoneme_segments, trim_silence
        )
        return self._kokoro._audio_generator._concatenate_audio_segments(processed)
