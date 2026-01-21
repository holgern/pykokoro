from __future__ import annotations

from typing import Any

import numpy as np

from ..generation_config import GenerationConfig
from ..pipeline import KokoroPipeline
from ..pipeline_config import PipelineConfig
from .warnings import maybe_warn


def generate(
    text: str,
    *,
    voice: str = "af",
    lang: str = "en-us",
    speed: float = 1.0,
    enable_deprecation_warnings: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Legacy API shim.

    Returns the raw audio array for backward compatibility.
    """
    generation_keys = {
        "pause_mode",
        "pause_clause",
        "pause_sentence",
        "pause_paragraph",
        "pause_variance",
        "random_seed",
        "enable_short_sentence",
        "is_phonemes",
    }
    generation_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in generation_keys}
    cfg = PipelineConfig(
        mode="compat",
        voice=voice,
        generation=GenerationConfig(speed=speed, lang=lang, **generation_kwargs),
        enable_deprecation_warnings=enable_deprecation_warnings,
    )
    maybe_warn(
        cfg,
        "pykokoro.compat.v0_api.generate() is deprecated; use "
        "KokoroPipeline(PipelineConfig(...)).run(text)",
    )
    pipe = KokoroPipeline(cfg)
    res = pipe.run(text)
    return res.audio
