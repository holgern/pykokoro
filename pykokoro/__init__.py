"""PyKokoro - pipeline-only API for Kokoro TTS."""

from .pipeline import KokoroPipeline
from .pipeline_config import PipelineConfig
from .generation_config import GenerationConfig

# Version info
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

__all__ = [
    "__version__",
    "KokoroPipeline",
    "PipelineConfig",
    "GenerationConfig",
]
