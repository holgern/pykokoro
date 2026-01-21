from __future__ import annotations

from typing import Any

import numpy as np

from ...config import PipelineConfig
from ...types import PhonemeSegment, Segment


class OnnxSynthesizer:
    """Stub ONNX synthesizer.

    Wire this to your existing ONNX backend implementation.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def synth(self, items: list[PhonemeSegment], *, segments: list[Segment]) -> tuple[np.ndarray, int]:
        # TODO: integrate your existing ONNX backend.
        # For now, return 0.1s of silence at 24kHz.
        sr = 24000
        return np.zeros(int(0.1 * sr), dtype=np.float32), sr
