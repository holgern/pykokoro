"""PyKokoro - Python library for Kokoro TTS.

This library provides ONNX-based text-to-speech using the Kokoro model,
with support for multiple languages, voices, and quality levels.
"""

from .constants import DEFAULT_CONFIG, PROGRAM_NAME
from .generation_config import GenerationConfig
from .onnx_backend import (
    Kokoro,
    ModelQuality,
    VoiceBlend,
    download_all_models,
    download_all_voices,
    download_config,
    download_model,
    download_voice,
    get_model_path,
    get_voice_path,
    is_chinese_language,
    load_vocab_from_config,
)
from .phonemes import (
    PhonemeSegment,
    phonemize_text_list,
    split_and_phonemize_text,
    text_to_phoneme_segments,
)
from .ssmd_parser import SSMDMetadata, SSMDSegment
from .tokenizer import (
    EspeakConfig,
    PhonemeResult,
    Tokenizer,
    TokenizerConfig,
    create_tokenizer,
)
from .trim import trim
from .utils import (
    get_device,
    get_gpu_info,
    get_user_cache_path,
    get_user_config_path,
    load_config,
    save_config,
)

# Version info
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

__all__ = [
    # Version
    "__version__",
    # Constants
    "PROGRAM_NAME",
    "DEFAULT_CONFIG",
    # Main classes
    "Kokoro",
    "Tokenizer",
    "VoiceBlend",
    # Phoneme classes
    "PhonemeSegment",
    # SSMD classes
    "SSMDMetadata",
    "SSMDSegment",
    # Config classes
    "GenerationConfig",
    "TokenizerConfig",
    "EspeakConfig",
    "PhonemeResult",
    "ModelQuality",
    # Download functions
    "download_model",
    "download_voice",
    "download_all_voices",
    "download_all_models",
    "download_config",
    # Path functions
    "get_model_path",
    "get_voice_path",
    "get_user_cache_path",
    "get_user_config_path",
    # Config functions
    "load_config",
    "save_config",
    # Device functions
    "get_device",
    "get_gpu_info",
    # Audio functions
    "trim",
    # Helper functions
    "create_tokenizer",
    "phonemize_text_list",
    "split_and_phonemize_text",
    "text_to_phoneme_segments",
    "is_chinese_language",
    "load_vocab_from_config",
]

# Re-export Document from ssmd
from ssmd import Document

__all__.append("Document")
