from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing configuration for the end-to-end pipeline.

    Keep this frozen+hashable so it can be used as part of cache keys.
    """

    voice: str = "af"
    lang: str = "en"

    speed: float = 1.0
    max_chars: int | None = 400

    # Stage selection
    split: Literal["auto", "phrasplit", "none"] = "auto"
    markup: Literal["auto", "ssmd", "none"] = "auto"

    # Behavior toggles
    return_trace: bool = False
    legacy_alignment: bool = False

    # Caching
    cache_dir: str | None = None
