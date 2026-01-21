from __future__ import annotations

import warnings

from ..pipeline_config import PipelineConfig


def maybe_warn(cfg: PipelineConfig, message: str) -> None:
    if cfg.enable_deprecation_warnings:
        warnings.warn(message, DeprecationWarning, stacklevel=3)
