from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..pipeline import KokoroPipeline


def generate(text: str, *, voice: str = "af", lang: str = "en", speed: float = 1.0, **kwargs: Any) -> np.ndarray:
    """Legacy API shim.

    Returns the raw audio array for backward compatibility.
    """
    warnings.warn(
        "pykokoro.compat.v0_api.generate() is deprecated; use KokoroPipeline(PipelineConfig(...)).run(text)",
        DeprecationWarning,
        stacklevel=2,
    )
    pipe = KokoroPipeline(PipelineConfig(voice=voice, lang=lang, speed=speed))
    res = pipe.run(text)
    return res.audio
