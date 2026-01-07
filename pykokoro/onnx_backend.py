"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

import asyncio
import io
import logging
import os
import re
import sqlite3
import urllib.request
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download

from .phonemes import PhonemeSegment
from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .trim import trim as trim_audio
from .utils import get_user_cache_path

# Logger for debugging
logger = logging.getLogger(__name__)

# Maximum phoneme length for a single inference
MAX_PHONEME_LENGTH = 510

# Sample rate for Kokoro models
SAMPLE_RATE = 24000

# Model quality type
ModelQuality = Literal[
    "fp32", "fp16", "fp16-gpu", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]
DEFAULT_MODEL_QUALITY: ModelQuality = "fp32"

# Provider type
ProviderType = Literal["auto", "cpu", "cuda", "openvino", "directml", "coreml"]

# Model source type
ModelSource = Literal["huggingface", "github"]
DEFAULT_MODEL_SOURCE: ModelSource = "huggingface"

# Model variant type (for GitHub source)
ModelVariant = Literal["v1.0", "v1.1-zh"]
DEFAULT_MODEL_VARIANT: ModelVariant = "v1.0"

# Quality to filename mapping (Hugging Face)
MODEL_QUALITY_FILES_HF: dict[str, str] = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q8": "model_quantized.onnx",
    "q8f16": "model_q8f16.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
    "uint8": "model_uint8.onnx",
    "uint8f16": "model_uint8f16.onnx",
}

# Quality to filename mapping (GitHub v1.0 - English)
MODEL_QUALITY_FILES_GITHUB_V1_0: dict[str, str] = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "fp16-gpu": "kokoro-v1.0.fp16-gpu.onnx",
    "q8": "kokoro-v1.0.int8.onnx",
}

# Quality to filename mapping (GitHub v1.1-zh - Chinese)
MODEL_QUALITY_FILES_GITHUB_V1_1_ZH: dict[str, str] = {
    "fp32": "kokoro-v1.1-zh.onnx",
}

# Backward compatibility
MODEL_QUALITY_FILES = MODEL_QUALITY_FILES_HF

# URLs for model files (Hugging Face)
HF_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
HF_MODEL_SUBFOLDER = "onnx"
HF_VOICES_SUBFOLDER = "voices"
HF_CONFIG_FILENAME = "config.json"

# HuggingFace repositories for different model variants
HF_REPO_V1_0 = "hexgrad/Kokoro-82M"  # English/multilingual v1.0
HF_REPO_V1_1_ZH = "hexgrad/Kokoro-82M-v1.1-zh"  # Chinese v1.1-zh

# Config filenames with variant suffix (for local storage)
HF_CONFIG_FILENAME_V1_1_ZH = "config-v1.1-zh.json"  # Variant suffix format

# URLs for model files (GitHub)
GITHUB_REPO = "thewh1teagle/kokoro-onnx"

# GitHub v1.0 (English)
GITHUB_RELEASE_TAG_V1_0 = "model-files-v1.0"
GITHUB_BASE_URL_V1_0 = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG_V1_0}"
)
GITHUB_VOICES_FILENAME_V1_0 = "voices-v1.0.bin"

# GitHub v1.1-zh (Chinese)
GITHUB_RELEASE_TAG_V1_1_ZH = "model-files-v1.1"
GITHUB_BASE_URL_V1_1_ZH = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG_V1_1_ZH}"
)
GITHUB_VOICES_FILENAME_V1_1_ZH = "voices-v1.1-zh.bin"

# Backward compatibility
GITHUB_RELEASE_TAG = GITHUB_RELEASE_TAG_V1_0
GITHUB_BASE_URL = GITHUB_BASE_URL_V1_0
GITHUB_VOICES_FILENAME = GITHUB_VOICES_FILENAME_V1_0

# All available voice names (HuggingFace and GitHub v1.0 - English)
# These are used for downloading individual voice files from HuggingFace
VOICE_NAMES = [
    "af",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]

# Expected voice names for GitHub v1.1-zh (Chinese model)
# Note: These are loaded dynamically from voices.bin, this list is for reference
# The v1.1-zh model contains 103 voices with various Chinese speakers
VOICE_NAMES_ZH = [
    # Sample voices from the v1.1-zh model:
    "af_maple",  # Female voice
    "af_sol",  # Female voice
    "bf_vale",  # British female voice
    # Numbered Chinese female voices (zf_XXX)
    "zf_001",
    "zf_002",
    "zf_003",  # ... many more numbered voices
    # Numbered Chinese male voices (zm_XXX)
    "zm_009",
    "zm_010",
    "zm_011",  # ... many more numbered voices
    # Note: Full list contains 103 voices total
    # Use kokoro.get_voices() to retrieve the complete list at runtime
]

# Voice name documentation by language/variant
# These voices are dynamically loaded from the model's voices.bin file
# The actual available voices may vary depending on the model source and variant
VOICE_NAMES_BY_VARIANT = {
    "huggingface": VOICE_NAMES,  # All voices (multi-language)
    "github-v1.0": VOICE_NAMES,  # Same as HuggingFace (multi-language)
    "github-v1.1-zh": VOICE_NAMES_ZH,  # Chinese-specific voices
}


@dataclass
class VoiceBlend:
    """Represents a blend of multiple voices."""

    voices: list[tuple[str, float]]  # List of (voice_name, weight) tuples

    @classmethod
    def parse(cls, blend_str: str) -> "VoiceBlend":
        """
        Parse a voice blend string.

        Format: "voice1:weight1,voice2:weight2" or "voice1:50,voice2:50"
        Weights should sum to 100 (percentages).

        Example: "af_nicole:50,am_michael:50"
        """
        voices = []
        for part in blend_str.split(","):
            part = part.strip()
            if ":" in part:
                voice_name, weight_str = part.split(":", 1)
                weight = float(weight_str) / 100.0  # Convert percentage to fraction
            else:
                voice_name = part
                weight = 1.0
            voices.append((voice_name.strip(), weight))

        # Normalize weights if they don't sum to 1
        total_weight = sum(w for _, w in voices)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            voices = [(v, w / total_weight) for v, w in voices]

        return cls(voices=voices)


# =============================================================================
# Path helper functions
# =============================================================================


def get_model_dir() -> Path:
    """Get the directory for storing ONNX model files."""
    return get_user_cache_path("models") / "onnx"


def get_voices_dir() -> Path:
    """Get the directory for storing voice files."""
    return get_user_cache_path("voices")


def get_config_path(variant: ModelVariant | None = None) -> Path:
    """Get the path to the cached config.json for a specific variant.

    Args:
        variant: Model variant ("v1.0", "v1.1-zh", or None for default)

    Returns:
        Path to config file in variant-specific subdirectory
    """
    if variant == "v1.1-zh":
        return get_user_cache_path() / "v1.1-zh" / HF_CONFIG_FILENAME
    else:
        # v1.0 or None (default)
        return get_user_cache_path() / "v1.0" / HF_CONFIG_FILENAME


def get_voices_bin_path() -> Path:
    """Get the path to the combined voices.bin.npz file."""
    return get_user_cache_path() / "voices.bin.npz"


def get_model_filename(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> str:
    """Get the model filename for a quality level."""
    return MODEL_QUALITY_FILES[quality]


def get_model_path(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> Path:
    """Get the full path to a model file for a given quality."""
    filename = get_model_filename(quality)
    return get_model_dir() / filename


def get_voice_path(voice_name: str) -> Path:
    """Get the full path to an individual voice file."""
    return get_voices_dir() / f"{voice_name}.bin"


# =============================================================================
# Download check functions
# =============================================================================


def is_config_downloaded(variant: ModelVariant | None = None) -> bool:
    """Check if config.json is downloaded for a specific variant.

    Args:
        variant: Model variant ("v1.0", "v1.1-zh", or None for default)

    Returns:
        True if config exists and has content, False otherwise
    """
    config_path = get_config_path(variant)
    return config_path.exists() and config_path.stat().st_size > 0


def is_model_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if a model file is already downloaded for a given quality."""
    model_path = get_model_path(quality)
    return model_path.exists() and model_path.stat().st_size > 0


def is_voice_downloaded(voice_name: str) -> bool:
    """Check if an individual voice file is already downloaded."""
    voice_path = get_voice_path(voice_name)
    return voice_path.exists() and voice_path.stat().st_size > 0


def are_voices_downloaded() -> bool:
    """Check if the combined voices.bin file exists."""
    voices_bin_path = get_voices_bin_path()
    return voices_bin_path.exists() and voices_bin_path.stat().st_size > 0


def are_models_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if model, config, and voices.bin are downloaded."""
    return (
        is_config_downloaded()
        and is_model_downloaded(quality)
        and are_voices_downloaded()
    )


# =============================================================================
# Download functions
# =============================================================================


def _download_from_hf(
    repo_id: str,
    filename: str,
    subfolder: str | None = None,
    local_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download a file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: File to download
        subfolder: Subfolder in the repository
        local_dir: Local directory to save to
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Use hf_hub_download to download the file
    # It handles caching automatically
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=str(local_dir) if local_dir else None,
        force_download=force,
    )
    return Path(downloaded_path)


def download_config(
    variant: ModelVariant | None = None,
    force: bool = False,
) -> Path:
    """Download config.json from Hugging Face for a specific variant.

    Args:
        variant: Model variant ("v1.0", "v1.1-zh", or None for default v1.0)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded config file
    """
    # Determine repo and local directory based on variant
    if variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
        local_dir = get_user_cache_path() / "v1.1-zh"
    else:  # v1.0 or None (default)
        repo_id = HF_REPO_V1_0
        local_dir = get_user_cache_path() / "v1.0"

    # Create the directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)

    return _download_from_hf(
        repo_id=repo_id,
        filename=HF_CONFIG_FILENAME,  # Always "config.json" in the repo
        local_dir=local_dir,
        force=force,
    )


def load_vocab_from_config(variant: ModelVariant | None = None) -> dict[str, int]:
    """Load vocabulary from variant-specific config.json.

    Args:
        variant: Model variant ("v1.0", "v1.1-zh", or None for default)

    Returns:
        Dictionary mapping phoneme characters to token indices

    Raises:
        FileNotFoundError: If config file doesn't exist after download
        ValueError: If config doesn't contain vocab
    """
    import json

    from kokorog2p import get_kokoro_vocab

    config_path = get_config_path(variant)

    # Download if not exists
    if not config_path.exists():
        logger.info(f"Downloading config for variant '{variant}'...")
        try:
            download_config(variant=variant)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download config for variant '{variant}': {e}"
            ) from e

    # Load config
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load config from {config_path}: {e}. "
            f"Falling back to default vocabulary."
        )
        return get_kokoro_vocab()

    # Extract vocabulary
    if "vocab" not in config:
        raise ValueError(
            f"Config at {config_path} does not contain 'vocab' key. "
            f"Cannot load variant-specific vocabulary."
        )

    vocab = config["vocab"]
    logger.info(
        f"Loaded vocabulary with {len(vocab)} tokens "
        f"for variant '{variant}' from {config_path.name}"
    )

    return vocab


def download_model(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download a model file for the specified quality.

    Args:
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file
    """
    filename = get_model_filename(quality)
    return _download_from_hf(
        repo_id=HF_REPO_ID,
        filename=filename,
        subfolder=HF_MODEL_SUBFOLDER,
        local_dir=get_model_dir(),
        force=force,
    )


def download_voice(
    voice_name: str,
    force: bool = False,
) -> Path:
    """
    Download a single voice file.

    Args:
        voice_name: Name of the voice to download
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voice file
    """
    filename = f"{voice_name}.bin"
    return _download_from_hf(
        repo_id=HF_REPO_ID,
        filename=filename,
        subfolder=HF_VOICES_SUBFOLDER,
        local_dir=get_voices_dir(),
        force=force,
    )


def download_all_voices(
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """
    Download all voice files and config, then combine into a single voices.bin file.

    Downloads individual voice .bin files and config.json from Hugging Face,
    loads them, and saves them as a combined numpy archive (voices.bin.npz) for
    efficient loading.

    Args:
        progress_callback: Optional callback (voice_name, current_index, total_count)
        force: Force re-download even if files exist

    Returns:
        Path to the combined voices.bin.npz file
    """
    voices_bin_path = get_voices_bin_path()

    # If voices.bin already exists and not forcing, skip download
    if voices_bin_path.exists() and not force:
        return voices_bin_path

    voices_dir = get_voices_dir()
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Download config.json first
    if progress_callback:
        progress_callback("config.json", 0, len(VOICE_NAMES) + 1)
    download_config(force=force)

    total = len(VOICE_NAMES)
    voices: dict[str, np.ndarray] = {}

    for idx, voice_name in enumerate(VOICE_NAMES):
        if progress_callback:
            progress_callback(voice_name, idx + 1, total)

        # Download individual voice file
        voice_path = download_voice(voice_name, force=force)

        # Load the voice data from .bin file
        voice_data = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
        voices[voice_name] = voice_data

    # Save all voices to a single .npz file using np.savez
    voices_bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(voices_bin_path), **voices)  # type: ignore[arg-type]

    return voices_bin_path


def download_all_models(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download config, model, and all voice files.

    Args:
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download config
    if progress_callback:
        progress_callback("config.json", 0, 0)
    paths["config.json"] = download_config(force=force)

    # Download model
    model_filename = get_model_filename(quality)
    if progress_callback:
        progress_callback(model_filename, 0, 0)
    paths[model_filename] = download_model(quality, force=force)

    # Download all voices and combine into voices.bin.npz
    paths["voices.bin.npz"] = download_all_voices(progress_callback, force)

    return paths


# ============================================================================
# GitHub Download Functions
# ============================================================================


def _download_from_github(
    url: str,
    local_path: Path,
    force: bool = False,
) -> Path:
    """
    Download a file from GitHub releases using urllib.

    Args:
        url: Full URL to the file
        local_path: Local path to save the file
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Check if file already exists
    if local_path.exists() and not force:
        logger.debug(f"File already exists: {local_path}")
        return local_path

    # Create parent directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from {url} to {local_path}")

    try:
        # Download the file
        with urllib.request.urlopen(url) as response:
            content = response.read()

        # Write to file
        with open(local_path, "wb") as f:
            f.write(content)

        logger.info(f"Downloaded {local_path.name} ({len(content)} bytes)")
        return local_path

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def download_model_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download a model file from GitHub releases.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If quality is not available for the variant
    """
    # Get the appropriate quality mapping and base URL
    if variant == "v1.0":
        quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
        base_url = GITHUB_BASE_URL_V1_0
    elif variant == "v1.1-zh":
        quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
        base_url = GITHUB_BASE_URL_V1_1_ZH
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    # Check if quality is available for this variant
    if quality not in quality_files:
        available = ", ".join(quality_files.keys())
        raise ValueError(
            f"Quality '{quality}' not available for variant '{variant}'. "
            f"Available qualities: {available}"
        )

    # Get filename and construct URL
    filename = quality_files[quality]
    url = f"{base_url}/{filename}"

    # Determine local path
    model_dir = get_model_dir()
    # Use variant-specific subdirectory to avoid conflicts
    if variant != "v1.0":
        model_dir = model_dir / variant
    local_path = model_dir / filename

    # Download
    return _download_from_github(url, local_path, force)


def download_voices_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download voices.bin file from GitHub releases.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voices.bin file
    """
    # Get the appropriate filename and base URL
    if variant == "v1.0":
        filename = GITHUB_VOICES_FILENAME_V1_0
        base_url = GITHUB_BASE_URL_V1_0
    elif variant == "v1.1-zh":
        filename = GITHUB_VOICES_FILENAME_V1_1_ZH
        base_url = GITHUB_BASE_URL_V1_1_ZH
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    # Construct URL
    url = f"{base_url}/{filename}"

    # Determine local path
    voices_dir = get_voices_dir()
    # Use variant-specific subdirectory to avoid conflicts
    if variant != "v1.0":
        voices_dir = voices_dir / variant
    local_path = voices_dir / filename

    # Download
    return _download_from_github(url, local_path, force)


def download_all_models_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download model and voices files from GitHub.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download model
    if progress_callback:
        progress_callback("model", 0, 2)
    model_path = download_model_github(variant, quality, force)
    paths[model_path.name] = model_path

    # Download voices
    if progress_callback:
        progress_callback("voices", 1, 2)
    voices_path = download_voices_github(variant, force)
    paths[voices_path.name] = voices_path

    if progress_callback:
        progress_callback("complete", 2, 2)

    return paths


class Kokoro:
    """
    Native ONNX backend for TTS generation.

    This class provides direct ONNX inference without external dependencies.
    Includes embedded tokenizer for phoneme/token-based generation.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        session_options: rt.SessionOptions | None = None,
        provider_options: dict[str, Any] | None = None,
        vocab_version: str = "v1.0",
        espeak_config: EspeakConfig | None = None,
        tokenizer_config: Optional["TokenizerConfig"] = None,
        model_quality: ModelQuality | None = None,
        model_source: ModelSource = DEFAULT_MODEL_SOURCE,
        model_variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    ) -> None:
        """
        Initialize the Kokoro ONNX backend.

        Args:
            model_path: Path to the ONNX model file (auto-downloaded if None)
            voices_path: Path to the voices.bin file (auto-downloaded if None)
            use_gpu: Deprecated. Use provider parameter instead.
                Legacy GPU flag for backward compatibility.
            provider: Execution provider for ONNX Runtime. Options:
                "auto" (auto-select best), "cpu", "cuda" (NVIDIA),
                "openvino" (Intel), "directml" (Windows), "coreml" (macOS)
            session_options: Pre-configured ONNX Runtime SessionOptions object.
                If provided, this takes precedence over provider_options.
                For advanced users who need full control over session configuration.
            provider_options: Dictionary of provider and session options.
                Supports both SessionOptions attributes and provider-specific options.

                Common SessionOptions attributes:
                - intra_op_num_threads: Parallelism within operations (default: auto)
                - inter_op_num_threads: Parallelism across operations (default: 1)
                - graph_optimization_level: 0-3 or GraphOptimizationLevel enum
                - execution_mode: Sequential or parallel
                - enable_profiling: Enable ONNX profiling

                Provider-specific options:

                OpenVINO:
                - device_type: "CPU_FP32", "GPU", etc.
                - precision: "FP32", "FP16", "BF16" (auto-set from model_quality)
                - num_of_threads: Number of threads (default: auto)
                - cache_dir: Model cache directory
                  (default: ~/.cache/pykokoro/openvino_cache)
                - enable_opencl_throttling: "true"/"false" for iGPU

                CUDA:
                - device_id: GPU device ID (default: 0)
                - gpu_mem_limit: Memory limit in bytes
                - arena_extend_strategy: "kNextPowerOfTwo", "kSameAsRequested"
                - cudnn_conv_algo_search: "EXHAUSTIVE", "HEURISTIC", "DEFAULT"

                DirectML:
                - device_id: GPU device ID
                - disable_metacommands: "true"/"false"

                CoreML:
                - MLComputeUnits: "ALL", "CPU_ONLY", "CPU_AND_GPU"
                - EnableOnSubgraphs: "true"/"false"

                Example:
                    provider_options={
                        "precision": "FP16",
                        "num_of_threads": 8,
                        "intra_op_num_threads": 4
                    }
            vocab_version: Vocabulary version for tokenizer
            espeak_config: Optional espeak-ng configuration
                (deprecated, use tokenizer_config)
            tokenizer_config: Optional tokenizer configuration
                (for mixed-language support)
            model_quality: Model quality/quantization level (default from config)
            model_source: Model source ("huggingface" or "github")
            model_variant: Model variant for GitHub source ("v1.0" or "v1.1-zh")
        """
        self._session: rt.InferenceSession | None = None
        self._voices_data: dict[str, np.ndarray] | None = None
        self._np = np

        # Deprecation warning for use_gpu
        if use_gpu:
            logger.warning(
                "The 'use_gpu' parameter is deprecated and will be removed in a "
                "future version. Use 'provider' parameter instead. "
                "Example: Kokoro(provider='cuda') or Kokoro(provider='auto')"
            )

        self._use_gpu = use_gpu
        self._provider: ProviderType | None = provider
        self._session_options = session_options
        self._model_source = model_source

        # Store initial variant (before auto-detection)
        self._initial_model_variant = model_variant
        self._model_variant = model_variant
        self._auto_switched_variant = False  # Track if we auto-switched

        # Load config for defaults
        from .utils import load_config

        cfg = load_config()

        # Resolve provider_options from config if not specified
        if provider_options is None and "provider_options" in cfg:
            provider_options = cfg.get("provider_options")
            logger.info(f"Loaded provider_options from config: {provider_options}")

        self._provider_options = provider_options

        # Resolve model quality from config if not specified
        resolved_quality: ModelQuality = DEFAULT_MODEL_QUALITY
        if model_quality is not None:
            resolved_quality = model_quality
        else:
            quality_from_cfg = cfg.get("model_quality", DEFAULT_MODEL_QUALITY)
            # Validate it's a valid quality option and cast to ModelQuality
            if quality_from_cfg in MODEL_QUALITY_FILES:
                resolved_quality = quality_from_cfg

        # Validate quality is available for the selected source/variant
        if model_source == "github":
            if model_variant == "v1.0":
                available_qualities = MODEL_QUALITY_FILES_GITHUB_V1_0
            elif model_variant == "v1.1-zh":
                available_qualities = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
            else:
                raise ValueError(f"Unknown model variant: {model_variant}")

            if resolved_quality not in available_qualities:
                available = ", ".join(available_qualities.keys())
                raise ValueError(
                    f"Quality '{resolved_quality}' not available for "
                    f"GitHub {model_variant}. Available qualities: {available}"
                )
        elif model_source == "huggingface":
            if resolved_quality not in MODEL_QUALITY_FILES_HF:
                available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
                raise ValueError(
                    f"Quality '{resolved_quality}' not available for HuggingFace. "
                    f"Available qualities: {available}"
                )

        self._model_quality: ModelQuality = resolved_quality

        # Resolve paths
        if model_path is None:
            if model_source == "github":
                # Download from GitHub if not exists
                model_path = download_model_github(
                    model_variant, self._model_quality, force=False
                )
            else:
                # Download from HuggingFace (default)
                model_path = get_model_path(self._model_quality)
        if voices_path is None:
            if model_source == "github":
                # Download from GitHub if not exists
                voices_path = download_voices_github(model_variant, force=False)
            else:
                # Download from HuggingFace (default)
                voices_path = get_voices_bin_path()

        self._model_path = model_path
        self._voices_path = voices_path

        # Voice database connection (for kokovoicelab integration)
        self._voice_db: sqlite3.Connection | None = None

        # Tokenizer for phoneme-based generation
        self._tokenizer: Tokenizer | None = None
        # Use model variant as vocab version for proper filtering
        self._vocab_version = self._model_variant
        self._espeak_config = espeak_config
        self._tokenizer_config = tokenizer_config

    def _get_vocabulary(self) -> dict[str, int]:
        """Get vocabulary for the current model variant.

        Returns:
            Dictionary mapping phoneme characters to token indices
        """
        from kokorog2p import get_kokoro_vocab

        # For GitHub models, load variant-specific vocab from config
        if self._model_source == "github":
            return load_vocab_from_config(self._model_variant)

        # For HuggingFace or default, use standard vocab
        return get_kokoro_vocab()

    def _resolve_model_variant(self, lang: str) -> ModelVariant:
        """Resolve the appropriate model variant based on language.

        Automatically switches to v1.1-zh for Chinese languages unless
        user explicitly specified a variant.

        Args:
            lang: Language code for the text being synthesized

        Returns:
            Resolved model variant to use
        """
        # If user explicitly specified variant, don't auto-switch
        # (Check if variant differs from default)
        if self._initial_model_variant != DEFAULT_MODEL_VARIANT:
            return self._model_variant

        # Auto-detect: Switch to v1.1-zh for Chinese
        if is_chinese_language(lang) and self._model_source == "github":
            if not self._auto_switched_variant:
                logger.info(
                    f"Detected Chinese language '{lang}'. "
                    f"Automatically switching to model variant 'v1.1-zh'."
                )
                self._auto_switched_variant = True
            return "v1.1-zh"

        # Otherwise use configured variant
        return self._model_variant

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance (lazily initialized).

        Uses variant-specific vocabulary for proper phoneme filtering.
        """
        if self._tokenizer is None:
            # Get variant-specific vocabulary
            vocab = self._get_vocabulary()

            logger.debug(
                f"Initializing tokenizer with {len(vocab)} tokens "
                f"for variant '{self._model_variant}'"
            )

            self._tokenizer = Tokenizer(
                config=self._tokenizer_config,
                espeak_config=self._espeak_config,
                vocab_version=self._vocab_version,
                vocab=vocab,  # Pass variant-specific vocabulary
            )
        return self._tokenizer

    def _ensure_models(self) -> None:
        """Ensure model, voice, and config files are downloaded for current variant."""
        # Download model if needed
        if not self._model_path.exists():
            if self._model_source == "github":
                download_model_github(
                    variant=self._model_variant, quality=self._model_quality
                )
            else:  # huggingface
                download_model(self._model_quality)

        # Download voices if needed
        if not self._voices_path.exists():
            if self._model_source == "github":
                download_voices_github(variant=self._model_variant)
            else:  # huggingface
                download_all_voices()

        # Download variant-specific config if needed
        if self._model_source == "github":
            if not is_config_downloaded(variant=self._model_variant):
                logger.info(
                    f"Downloading config for variant '{self._model_variant}'..."
                )
                download_config(variant=self._model_variant)
        else:  # huggingface - default v1.0
            if not is_config_downloaded():
                download_config()

    def _get_default_provider_options(self, provider: str) -> dict[str, str]:
        """
        Get sensible default options for a provider.

        Uses PyKokoro cache path and model quality for smart defaults.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")

        Returns:
            Dictionary of default provider options (string values)
        """
        defaults: dict[str, str] = {}

        if provider == "OpenVINOExecutionProvider":
            # Use cache dir from PyKokoro
            cache_dir = get_user_cache_path() / "openvino_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            defaults = {
                "device_type": "CPU_FP32",
                "cache_dir": str(cache_dir),
                "enable_opencl_throttling": "false",
            }

            # Auto-set precision based on model_quality
            if self._model_quality in ["fp16", "fp16-gpu"]:
                defaults["precision"] = "FP16"
            elif self._model_quality == "fp32":
                defaults["precision"] = "FP32"
            else:
                # For quantized models, use FP32 precision in OpenVINO
                defaults["precision"] = "FP32"

        elif provider == "CUDAExecutionProvider":
            defaults = {
                "device_id": "0",
                "arena_extend_strategy": "kNextPowerOfTwo",
            }

        elif provider == "DmlExecutionProvider":
            defaults = {
                "device_id": "0",
            }

        return defaults

    def _get_provider_specific_options(
        self,
        provider: str,
        all_options: dict[str, Any],
    ) -> dict[str, str]:
        """
        Extract provider-specific options for the given provider.

        Filters out SessionOptions attributes and converts values to strings
        as required by ONNX Runtime.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")
            all_options: Dictionary of all options (mixed session and provider options)

        Returns:
            Dictionary of provider-specific options with string values
        """
        # Define known provider options
        provider_options_map: dict[str, list[str]] = {
            "OpenVINOExecutionProvider": [
                "device_type",
                "precision",
                "num_of_threads",
                "cache_dir",
                "enable_opencl_throttling",
            ],
            "CUDAExecutionProvider": [
                "device_id",
                "gpu_mem_limit",
                "arena_extend_strategy",
                "cudnn_conv_algo_search",
                "do_copy_in_default_stream",
            ],
            "DmlExecutionProvider": ["device_id", "disable_metacommands"],
            "CoreMLExecutionProvider": [
                "MLComputeUnits",
                "EnableOnSubgraphs",
                "ModelFormat",
            ],
        }

        known_options = provider_options_map.get(provider, [])

        # General SessionOptions attributes (exclude from provider options)
        session_attrs = {
            "intra_op_num_threads",
            "inter_op_num_threads",
            "num_threads",
            "threads",
            "graph_optimization_level",
            "execution_mode",
            "enable_profiling",
            "enable_mem_pattern",
            "enable_cpu_mem_arena",
            "enable_mem_reuse",
            "log_severity_level",
            "log_verbosity_level",
        }

        # Extract only provider-specific options
        provider_opts: dict[str, str] = {}
        for key, value in all_options.items():
            if key in session_attrs:
                continue  # Skip SessionOptions attributes

            if known_options and key not in known_options:
                logger.warning(
                    f"Unknown option '{key}' for {provider}. "
                    f"Known options: {known_options}"
                )
                continue

            # Convert to string as required by ONNX Runtime
            provider_opts[key] = str(value)

        return provider_opts

    def _apply_provider_options(
        self,
        sess_opt: rt.SessionOptions,
        options: dict[str, Any],
    ) -> None:
        """
        Apply provider options to SessionOptions.

        Handles both SessionOptions attributes and provider-specific configs.

        Args:
            sess_opt: SessionOptions to modify
            options: Dictionary of options to apply
        """
        # Map of common option names to SessionOptions attributes
        session_option_attrs: dict[str, str] = {
            "intra_op_num_threads": "intra_op_num_threads",
            "inter_op_num_threads": "inter_op_num_threads",
            "num_threads": "intra_op_num_threads",  # Alias
            "threads": "intra_op_num_threads",  # Alias
            "graph_optimization_level": "graph_optimization_level",
            "execution_mode": "execution_mode",
            "enable_profiling": "enable_profiling",
            "enable_mem_pattern": "enable_mem_pattern",
            "enable_cpu_mem_arena": "enable_cpu_mem_arena",
            "enable_mem_reuse": "enable_mem_reuse",
            "log_severity_level": "log_severity_level",
            "log_verbosity_level": "log_verbosity_level",
        }

        # Apply SessionOptions attributes
        for opt_name, value in options.items():
            if opt_name in session_option_attrs:
                attr_name = session_option_attrs[opt_name]
                setattr(sess_opt, attr_name, value)
                logger.debug(f"Set SessionOptions.{attr_name} = {value}")

    def _create_session_options(self) -> rt.SessionOptions:
        """
        Create SessionOptions with user configuration and sensible defaults.

        Priority:
        1. User-provided SessionOptions object (self._session_options)
        2. User-provided provider_options dict (self._provider_options)
        3. Sensible defaults

        Returns:
            Configured SessionOptions instance
        """
        # If user provided a SessionOptions object, use it directly
        if self._session_options is not None:
            logger.info("Using user-provided SessionOptions")
            return self._session_options

        # Create new SessionOptions with defaults
        sess_opt = rt.SessionOptions()

        # Sensible defaults - let ONNX Runtime decide thread count
        # Only set these if user doesn't override
        sess_opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        # Apply user provider_options if provided
        if self._provider_options:
            logger.info(f"Applying provider options: {self._provider_options}")
            self._apply_provider_options(sess_opt, self._provider_options)

        return sess_opt

    def _select_providers(
        self,
        provider: ProviderType | None,
        use_gpu: bool,
    ) -> list[str | tuple[str, dict[str, str]]]:
        """
        Select ONNX Runtime execution providers based on preference.

        Args:
            provider: Explicit provider ('auto', 'cpu', 'cuda', 'openvino', etc.)
            use_gpu: Legacy GPU flag (for backward compatibility)

        Returns:
            List of providers in priority order. Can be simple strings or
            tuples of (provider_name, options_dict) for provider-specific options.

        Raises:
            RuntimeError: If requested provider is not available
            ValueError: If provider name is invalid
        """
        available = rt.get_available_providers()

        # Helper function to create provider list with options
        def _make_provider_list(prov: str) -> list[str | tuple[str, dict[str, str]]]:
            """Create provider list, adding options if needed."""
            # Get default options for this provider
            default_opts = self._get_default_provider_options(prov)

            # Get user-provided provider-specific options
            provider_opts = {}
            if self._provider_options:
                provider_opts = self._get_provider_specific_options(
                    prov, self._provider_options
                )

            # Merge defaults with user options (user options take precedence)
            merged_opts = {**default_opts, **provider_opts}

            if merged_opts:
                logger.info(f"Using {prov} with options: {merged_opts}")
                return [(prov, merged_opts), "CPUExecutionProvider"]
            else:
                return [prov, "CPUExecutionProvider"]

        # Environment variable override (highest priority)
        env_provider = os.getenv("ONNX_PROVIDER")
        if env_provider:
            logger.info(f"Using provider from ONNX_PROVIDER env: {env_provider}")
            return _make_provider_list(env_provider)

        # Auto-selection logic
        if provider == "auto" or (provider is None and use_gpu):
            # Priority: CUDA > OpenVINO > CoreML > DirectML
            for prov in [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            ]:
                if prov in available:
                    logger.info(f"Auto-selected provider: {prov}")
                    return _make_provider_list(prov)
            logger.info("Auto-selection: No accelerators found, using CPU")
            return ["CPUExecutionProvider"]

        # Default to CPU if no provider specified and use_gpu=False
        if provider is None:
            logger.info("Using CPU provider")
            return ["CPUExecutionProvider"]

        # Explicit provider selection
        provider_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
            "directml": "DmlExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }

        selected = provider_map.get(provider.lower())
        if not selected:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Valid options: {list(provider_map.keys())}"
            )

        if selected not in available:
            # Provide helpful installation message
            install_hints = {
                "CUDAExecutionProvider": "pip install pykokoro[gpu]",
                "OpenVINOExecutionProvider": "pip install pykokoro[openvino]",
                "DmlExecutionProvider": "pip install pykokoro[directml]",
                "CoreMLExecutionProvider": "pip install pykokoro[coreml]",
            }
            hint = install_hints.get(
                selected, f"install the required package for {selected}"
            )
            raise RuntimeError(
                f"{provider.upper()} provider requested but not available.\n"
                f"Install with: {hint}\n"
                f"Available providers: {available}"
            )

        logger.info(f"Using explicitly selected provider: {selected}")
        return _make_provider_list(selected)

    def _init_kokoro(self) -> None:
        """Initialize the ONNX session and load voices."""
        if self._session is not None:
            return

        self._ensure_models()

        # Create session options
        sess_options = self._create_session_options()

        # Select execution providers
        providers = self._select_providers(self._provider, self._use_gpu)

        # Try to load ONNX model with automatic fallback
        # If the primary provider fails and we're in auto mode, try CPU
        session_loaded = False
        last_error = None

        for attempt, provider_list in enumerate([providers, ["CPUExecutionProvider"]]):
            # Skip second attempt if we already tried CPU or
            # if explicit provider was requested
            if attempt == 1:
                if providers == ["CPUExecutionProvider"]:
                    break  # Already tried CPU
                if self._provider and self._provider != "auto":
                    break  # User explicitly requested a provider, don't fallback
                # Check if primary provider was already CPU (handle both str and tuple)
                primary_provider = providers[0]
                if isinstance(primary_provider, tuple):
                    primary_provider = primary_provider[0]
                if primary_provider == "CPUExecutionProvider":
                    break  # Primary was already CPU

            try:
                self._session = rt.InferenceSession(
                    str(self._model_path),
                    sess_options=sess_options,
                    providers=provider_list,
                )
                session_loaded = True

                # Log what was actually loaded
                actual_providers = self._session.get_providers()
                logger.info(f"Loaded ONNX session with providers: {actual_providers}")

                # Warn if we had to fallback
                if attempt == 1:
                    failed_provider = providers[0]
                    if isinstance(failed_provider, tuple):
                        failed_provider = failed_provider[0]
                    logger.warning(
                        f"Failed to load model with {failed_provider}, "
                        f"fell back to CPU. Error: {last_error}"
                    )

                break

            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    # First attempt failed, will try fallback
                    provider_name = provider_list[0]
                    if isinstance(provider_name, tuple):
                        provider_name = provider_name[0]
                    logger.debug(f"Provider {provider_name} failed: {e}")
                    continue
                else:
                    # Fallback also failed, re-raise
                    raise

        if not session_loaded:
            raise RuntimeError(
                f"Failed to initialize ONNX session with providers {providers}. "
                f"Last error: {last_error}"
            )

        # Load voices (numpy archive with voice style vectors or raw binary)
        if self._model_source == "github":
            # GitHub voices.bin format: raw binary file with voice data
            # We need to parse it to extract individual voices
            self._voices_data = self._load_voices_bin_github(self._voices_path)
        else:
            # HuggingFace format: .npz archive with named voice arrays
            self._voices_data = dict(np.load(str(self._voices_path), allow_pickle=True))

    def _load_voices_bin_github(self, voices_path: Path) -> dict[str, np.ndarray]:
        """
        Load voices from GitHub format .bin file.

        The GitHub voices.bin format is a NumPy archive file (.npz format)
        containing voice arrays with voice names as keys.

        Args:
            voices_path: Path to the voices.bin file

        Returns:
            Dictionary mapping voice names to numpy arrays
        """
        # Load the NumPy file - it's actually .npz format despite .bin extension
        voices_npz = np.load(str(voices_path), allow_pickle=True)

        # Convert NpzFile to dictionary
        voices: dict[str, np.ndarray] = dict(voices_npz)

        logger.info(f"Successfully loaded {len(voices)} voices from {voices_path}")
        logger.debug(f"Available voices: {', '.join(sorted(voices.keys()))}")

        return voices

    def _create_audio_internal(
        self,
        phonemes: str,
        voice: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """
        Core ONNX inference for a single phoneme batch.

        Args:
            phonemes: Phoneme string (will be truncated if > MAX_PHONEME_LENGTH)
            voice: Voice style vector
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio samples, sample rate)
        """
        assert self._session is not None

        # Truncate phonemes if too long
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        tokens = self.tokenizer.tokenize(phonemes)

        # Get voice style for this token length (clamp to valid range)
        style_idx = min(len(tokens), MAX_PHONEME_LENGTH - 1)
        voice_style = voice[style_idx]

        # Pad tokens with start/end tokens
        tokens_padded = [[0, *tokens, 0]]

        # Check input names to determine model version
        input_names = [i.name for i in self._session.get_inputs()]

        # GitHub models (v1.0 and v1.1-zh) use "input_ids" and int32 speed
        # HuggingFace newer models also use "input_ids" but with float32 speed
        if "input_ids" in input_names:
            # Check if this is a GitHub model by checking model source
            if self._model_source == "github":
                # GitHub models: input_ids, style (float32), speed (int32)
                speed_int = max(1, int(round(speed)))
                inputs = {
                    "input_ids": np.array(tokens_padded, dtype=np.int64),
                    "style": np.array(voice_style, dtype=np.float32),
                    "speed": np.array([speed_int], dtype=np.int32),
                }
            else:
                # HuggingFace original format: input_ids, float32 speed
                inputs = {
                    "input_ids": tokens_padded,
                    "style": voice_style,
                    "speed": np.ones(1, dtype=np.float32) * speed,
                }
        else:
            # Original model format (uses "tokens" input, float speed)
            inputs = {
                "tokens": tokens_padded,
                "style": voice_style,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        result = self._session.run(None, inputs)[0]
        audio = np.asarray(result).T
        # Ensure audio is 1D for compatibility with trim and other operations
        audio = np.squeeze(audio)
        return audio, SAMPLE_RATE

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """
        Split phonemes into batches at sentence-ending punctuation marks.

        Args:
            phonemes: Full phoneme string

        Returns:
            List of phoneme batches, each <= MAX_PHONEME_LENGTH
        """
        # Split on sentence-ending punctuation (., !, ?) while keeping them
        # Use lookbehind to split AFTER the punctuation
        sentences = re.split(r"(?<=[.!?])\s*", phonemes)

        batches = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding sentence would exceed limit, save current batch, start new
            if current and len(current) + len(sentence) + 1 > MAX_PHONEME_LENGTH:
                batches.append(current.strip())
                current = sentence
            # If the sentence itself is too long, we need to split it further
            elif len(sentence) > MAX_PHONEME_LENGTH:
                # Save current batch if any
                if current:
                    batches.append(current.strip())
                    current = ""
                # Split long sentence on any punctuation or spaces
                words = re.split(r"([.,;:!?\s])", sentence)
                # If there's no punctuation or spaces, force chunk by character count
                if len(words) == 1 and len(words[0]) > MAX_PHONEME_LENGTH:
                    # Chunk the string at MAX_PHONEME_LENGTH boundaries
                    chunk = words[0]
                    while len(chunk) > MAX_PHONEME_LENGTH:
                        batches.append(chunk[:MAX_PHONEME_LENGTH])
                        chunk = chunk[MAX_PHONEME_LENGTH:]
                    if chunk:
                        current = chunk
                else:
                    for word in words:
                        if not word or word.isspace():
                            if current:
                                current += " "
                            continue
                        if len(current) + len(word) + 1 > MAX_PHONEME_LENGTH:
                            if current:
                                batches.append(current.strip())
                            current = word
                        else:
                            if current and not current.endswith(
                                (".", "!", "?", ",", ";", ":")
                            ):
                                current += " "
                            current += word
            else:
                # Add sentence to current batch
                if current:
                    current += " "
                current += sentence

        if current:
            batches.append(current.strip())

        return batches if batches else [phonemes]

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        self._init_kokoro()
        assert self._voices_data is not None
        return list(sorted(self._voices_data.keys()))

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """
        Get the style vector for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Numpy array representing the voice style
        """
        self._init_kokoro()
        assert self._voices_data is not None
        return self._voices_data[voice_name]

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """
        Create a blended voice from multiple voices.

        Args:
            blend: VoiceBlend object specifying voices and weights

        Returns:
            Numpy array representing the blended voice style
        """
        self._init_kokoro()

        if len(blend.voices) == 1:
            voice_name, _ = blend.voices[0]
            return self.get_voice_style(voice_name)

        # Get style vectors and blend them
        blended: np.ndarray | None = None
        for voice_name, weight in blend.voices:
            style = self.get_voice_style(voice_name)
            weighted = style * weight
            if blended is None:
                blended = weighted
            else:
                blended = np.add(blended, weighted)

        # This should never be None if blend.voices is not empty
        assert blended is not None, "No voices in blend"
        return blended

    def _generate_from_phoneme_batches(
        self,
        batches: list[str],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Generate and concatenate audio from phoneme batches.

        Args:
            batches: List of phoneme strings (each <= MAX_PHONEME_LENGTH)
            voice_style: Voice style vector
            speed: Speech speed
            trim_silence: Whether to trim silence from each batch

        Returns:
            Concatenated audio array
        """
        audio_parts = []

        for batch in batches:
            audio, _ = self._create_audio_internal(batch, voice_style, speed)
            if trim_silence:
                audio, _ = trim_audio(audio)
            audio_parts.append(audio)

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def _generate_from_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Generate audio from list of PhonemeSegment instances.

        Unified audio generation method that handles:
        - Segments with phonemes (generate speech)
        - Empty segments (skip, only use pause_after)
        - Pause insertion based on pause_after field
        - Optional silence trimming

        Args:
            segments: List of PhonemeSegment instances
            voice_style: Voice style vector
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence from segment boundaries

        Returns:
            Concatenated audio array
        """
        from .utils import generate_silence

        audio_parts = []

        for segment in segments:
            # Generate speech if phonemes present
            if segment.phonemes.strip():
                # Handle long phonemes by splitting
                if len(segment.phonemes) > MAX_PHONEME_LENGTH:
                    batches = self._split_phonemes(segment.phonemes)
                    for batch in batches:
                        audio, _ = self._create_audio_internal(
                            batch, voice_style, speed
                        )
                        if trim_silence:
                            audio, _ = trim_audio(audio)
                        audio_parts.append(audio)
                else:
                    audio, _ = self._create_audio_internal(
                        segment.phonemes, voice_style, speed
                    )
                    if trim_silence:
                        audio, _ = trim_audio(audio)
                    audio_parts.append(audio)

            # Add pause after segment (works for both empty and non-empty phonemes)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def create(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        pause_short: float = 0.3,
        pause_medium: float = 0.6,
        pause_long: float = 1.0,
        pause_clause: float = 0.3,
        pause_sentence: float = 0.6,
        pause_paragraph: float = 1.0,
        split_mode: str | None = None,
        trim_silence: bool = False,
        pause_variance: float = 0.05,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text or phonemes.

        Pause markers (.), (..), (...) in text are automatically detected and
        processed as silence after the preceding segment.

        Args:
            text: Text to synthesize (or phonemes if is_phonemes=True). Pause
                markers (.), (..), (...) are automatically detected and converted
                to silence.
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb', 'es', 'fr')
            is_phonemes: If True, treat 'text' as phonemes instead of text
            pause_short: Duration for (.) in seconds, or clause pauses when
                trim_silence=True with split_mode
            pause_medium: Duration for (..) in seconds, or sentence pauses when
                trim_silence=True with split_mode
            pause_long: Duration for (...) in seconds, or paragraph pauses when
                trim_silence=True with split_mode
            pause_clause: Duration for clause pauses in seconds when
                trim_silence=True with split_mode
            pause_sentence: Duration for sentence pauses in seconds when
                trim_silence=True with split_mode
            pause_paragraph: Duration for paragraph pauses in seconds when
                trim_silence=True with split_mode
            split_mode: Optional text splitting mode. Options: None (default,
                automatic phoneme-based), "paragraph" (double newlines),
                "sentence" (requires spaCy), "clause" (sentences + commas,
                requires spaCy). When combined with trim_silence=True,
                automatically adds natural pauses between segments.
            trim_silence: Whether to trim silence from segment boundaries.
                When used with split_mode, adds natural pauses between
                segments (clause/sentence/paragraph boundaries).
            pause_variance: Standard deviation for Gaussian variance added to
                automatic pauses (in seconds). Only applies when trim_silence=True
                and split_mode is set. Default 0.05 (±100ms at 95% confidence).
                Set to 0.0 to disable variance.
            random_seed: Optional random seed for reproducible pause variance.
                If None, pauses will vary between runs.

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Auto-detect and switch variant if needed (e.g., for Chinese)
        resolved_variant = self._resolve_model_variant(lang)

        # If variant changed, we need to reinitialize
        if resolved_variant != self._model_variant:
            old_variant = self._model_variant
            self._model_variant = resolved_variant
            self._vocab_version = (
                resolved_variant  # Update vocab version to match variant
            )

            # Force re-initialization of resources for new variant
            self._tokenizer = None  # Tokenizer will reload with new vocab
            self._session = None  # Session will reload new model
            self._voices_data = None  # Voices will reload

            # Update paths for new variant
            if self._model_source == "github":
                # Update model path
                model_dir = get_model_dir()
                if resolved_variant != "v1.0":
                    model_dir = model_dir / resolved_variant

                if resolved_variant == "v1.0":
                    quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
                else:  # v1.1-zh
                    quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH

                filename = quality_files[self._model_quality]
                self._model_path = model_dir / filename

                # Update voices path
                voices_dir = get_voices_dir()
                if resolved_variant != "v1.0":
                    voices_dir = voices_dir / resolved_variant

                if resolved_variant == "v1.0":
                    voices_filename = GITHUB_VOICES_FILENAME_V1_0
                else:  # v1.1-zh
                    voices_filename = GITHUB_VOICES_FILENAME_V1_1_ZH

                self._voices_path = voices_dir / voices_filename

            # Ensure new variant files are downloaded
            self._ensure_models()

            # Re-initialize with new variant
            self._init_kokoro()

            logger.info(
                f"Switched from variant '{old_variant}' to '{resolved_variant}' "
                f"for language '{lang}'"
            )

        # Initialize random generator for reproducible variance
        rng = np.random.default_rng(random_seed)

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # If already phonemes, use directly
        if is_phonemes:
            phonemes = text
            batches = self._split_phonemes(phonemes)
            return self._generate_from_phoneme_batches(
                batches, voice_style, speed, trim_silence
            ), SAMPLE_RATE

        # Unified flow: text → segments → audio

        from .phonemes import text_to_phoneme_segments

        segments = text_to_phoneme_segments(
            text=text,
            tokenizer=self.tokenizer,
            lang=lang,
            split_mode=split_mode,
            pause_short=pause_short,
            pause_medium=pause_medium,
            pause_long=pause_long,
            pause_clause=pause_clause,
            pause_sentence=pause_sentence,
            pause_paragraph=pause_paragraph,
            pause_variance=pause_variance,
            trim_silence=trim_silence,
            rng=rng,
        )

        # Generate audio from segments
        audio = self._generate_from_segments(segments, voice_style, speed, trim_silence)

        return audio, SAMPLE_RATE

    def create_from_phonemes(
        self,
        phonemes: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from phonemes directly.

        This bypasses text-to-phoneme conversion, useful when working
        with pre-tokenized phoneme content.

        Args:
            phonemes: IPA phoneme string
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector if needed
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # kokoro-onnx supports direct phoneme input via create method with ps parameter
        # But we need to convert to tokens first
        tokens = self.tokenizer.tokenize(phonemes)

        # Debug logging for phoneme generation
        if os.getenv("TTSFORGE_DEBUG_PHONEMES"):
            logger.info(f"Phonemes: {phonemes}")
            logger.info(f"Tokens: {tokens}")

        return self.create_from_tokens(tokens, voice_style, speed)

    def create_from_tokens(
        self,
        tokens: list[int],
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from token IDs directly.

        This provides the lowest-level interface, useful for pre-tokenized
        content and maximum control.

        Args:
            tokens: List of token IDs
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Detokenize to phonemes and generate audio
        phonemes = self.tokenizer.detokenize(tokens)

        # Split phonemes into batches and generate audio
        batches = self._split_phonemes(phonemes)
        audio_parts = []

        for batch in batches:
            audio_part, _ = self._create_audio_internal(batch, voice_style, speed)
            # Trim silence from each part
            audio_part, _ = trim_audio(audio_part)
            audio_parts.append(audio_part)

        if not audio_parts:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        return np.concatenate(audio_parts), SAMPLE_RATE

    def create_from_segment(
        self,
        segment: "PhonemeSegment",
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str | None = None,
        trim_silence: bool = False,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from a PhonemeSegment.

        Respects the pause_after field by appending silence to the generated audio.

        Args:
            segment: PhonemeSegment with phonemes and tokens
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code override (e.g., 'de', 'en-us')
            trim_silence: Whether to trim silence from segment boundaries

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        from .utils import generate_silence

        # Debug logging for segment
        if os.getenv("TTSFORGE_DEBUG_PHONEMES"):
            logger.info(f"Segment text: {segment.text[:100]}...")
            logger.info(f"Segment phonemes: {segment.phonemes}")
            logger.info(f"Segment tokens: {segment.tokens}")
            logger.info(f"Segment lang: {segment.lang}")
            logger.info(f"Segment pause_after: {segment.pause_after}")

        # Generate audio for the segment
        # Use tokens if available, otherwise use phonemes
        if segment.tokens:
            audio, sample_rate = self.create_from_tokens(segment.tokens, voice, speed)
        elif segment.phonemes:
            audio, sample_rate = self.create_from_phonemes(
                segment.phonemes, voice, speed
            )
        else:
            # Fall back to text
            # Use lang override if provided, otherwise use segment's lang
            effective_lang = lang if lang is not None else segment.lang
            audio, sample_rate = self.create(segment.text, voice, speed, effective_lang)
        if trim_silence:
            audio, _ = trim_audio(audio)
        # Add pause after segment if specified
        if segment.pause_after > 0:
            pause_audio = generate_silence(segment.pause_after, sample_rate)
            audio = np.concatenate([audio, pause_audio])

        return audio, sample_rate

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        """
        Convert text to phonemes.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Phoneme string
        """
        return self.tokenizer.phonemize(text, lang=lang)

    def tokenize(self, phonemes: str) -> list[int]:
        """
        Convert phonemes to tokens.

        Args:
            phonemes: Phoneme string

        Returns:
            List of token IDs
        """
        return self.tokenizer.tokenize(phonemes)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert tokens back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return self.tokenizer.detokenize(tokens)

    def text_to_tokens(self, text: str, lang: str = "en-us") -> list[int]:
        """
        Convert text directly to tokens.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of token IDs
        """
        return self.tokenizer.text_to_tokens(text, lang=lang)

    def generate_chunks(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 500,
    ) -> Generator[tuple[np.ndarray, int, str], None, None]:
        """
        Generate audio in chunks for long text.

        This splits text into manageable chunks and yields audio for each.
        Useful for progress tracking during long conversions.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed
            lang: Language code
            chunk_size: Approximate character count per chunk

        Yields:
            Tuple of (audio samples, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice once
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Split text into chunks at sentence boundaries
        chunks = self._split_text(text, chunk_size)

        for chunk in chunks:
            if not chunk.strip():
                continue

            # Convert chunk to phonemes and generate audio
            phonemes = self.tokenizer.phonemize(chunk, lang=lang)
            batches = self._split_phonemes(phonemes)
            audio_parts = []

            for batch in batches:
                audio_part, _ = self._create_audio_internal(batch, voice_style, speed)
                audio_part, _ = trim_audio(audio_part)
                audio_parts.append(audio_part)

            if audio_parts:
                samples = np.concatenate(audio_parts)
                yield samples, SAMPLE_RATE, chunk

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        # Split on sentence boundaries while keeping the delimiter
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Voice Database Integration (from kokovoicelab)

    def load_voice_database(self, db_path: Path) -> None:
        """
        Load a voice database for custom/synthetic voices.

        Args:
            db_path: Path to the SQLite voice database
        """
        if self._voice_db is not None:
            self._voice_db.close()

        # Register numpy array converter
        sqlite3.register_converter("array", self._convert_array)
        self._voice_db = sqlite3.connect(
            str(db_path), detect_types=sqlite3.PARSE_DECLTYPES
        )

    def _convert_array(self, blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy array."""
        out = io.BytesIO(blob)
        return np.load(out)

    def get_voice_from_database(self, voice_name: str) -> np.ndarray | None:
        """
        Get a voice style vector from the database.

        Args:
            voice_name: Name of the voice in the database

        Returns:
            Voice style vector or None if not found
        """
        if self._voice_db is None:
            return None

        cursor = self._voice_db.cursor()
        cursor.execute(
            "SELECT style_vector FROM voices WHERE name = ?",
            (voice_name,),
        )
        row = cursor.fetchone()

        if row:
            return row[0]
        return None

    def list_database_voices(self) -> list[dict[str, Any]]:
        """
        List all voices in the database.

        Returns:
            List of voice metadata dictionaries
        """
        if self._voice_db is None:
            return []

        cursor = self._voice_db.cursor()
        cursor.execute(
            """
            SELECT name, gender, language, quality, is_synthetic, notes
            FROM voices
            ORDER BY quality DESC
            """
        )

        voices = []
        for row in cursor.fetchall():
            voices.append(
                {
                    "name": row[0],
                    "gender": row[1],
                    "language": row[2],
                    "quality": row[3],
                    "is_synthetic": bool(row[4]),
                    "notes": row[5],
                }
            )

        return voices

    def interpolate_voices(
        self,
        voice1: str | np.ndarray,
        voice2: str | np.ndarray,
        factor: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two voices.

        This uses the interpolation method from kokovoicelab to create
        voices that lie on the line between two source voices.

        Args:
            voice1: First voice (name or style vector)
            voice2: Second voice (name or style vector)
            factor: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice style vector
        """
        self._init_kokoro()

        # Resolve to style vectors
        if isinstance(voice1, str):
            style1 = self.get_voice_style(voice1)
        else:
            style1 = voice1

        if isinstance(voice2, str):
            style2 = self.get_voice_style(voice2)
        else:
            style2 = voice2

        # Use kokovoicelab's interpolation method
        diff_vector = style2 - style1
        midpoint = (style1 + style2) / 2
        return midpoint + (diff_vector * factor / 2)

    async def create_stream(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> AsyncGenerator[tuple[np.ndarray, int, str], None]:
        """
        Stream audio creation asynchronously, yielding chunks as they are processed.

        This method generates audio in the background and yields chunks as soon as
        they're ready, enabling real-time playback while generation continues.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb')

        Yields:
            Tuple of (audio samples as numpy array, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Convert text to phonemes
        phonemes = self.tokenizer.phonemize(text, lang=lang)

        # Split phonemes into batches
        batched_phonemes = self._split_phonemes(phonemes)

        # Create a queue for passing audio chunks
        queue: asyncio.Queue[tuple[np.ndarray, int, str] | None] = asyncio.Queue()

        async def process_batches() -> None:
            """Process phoneme batches in the background."""
            loop = asyncio.get_event_loop()
            for phoneme_batch in batched_phonemes:
                # Execute blocking ONNX inference in a thread executor
                audio_part, sample_rate = await loop.run_in_executor(
                    None, self._create_audio_internal, phoneme_batch, voice_style, speed
                )
                # Trim silence
                audio_part, _ = trim_audio(audio_part)
                await queue.put((audio_part, sample_rate, phoneme_batch))
            await queue.put(None)  # Signal end of stream

        # Start processing in the background
        asyncio.create_task(process_batches())

        # Yield chunks as they become available
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def create_stream_sync(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> Generator[tuple[np.ndarray, int, str], None, None]:
        """
        Stream audio creation synchronously, yielding chunks as they are processed.

        This is a synchronous version of create_stream for use in non-async contexts.
        It yields audio chunks immediately as they're generated.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb')

        Yields:
            Tuple of (audio samples as numpy array, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Convert text to phonemes
        phonemes = self.tokenizer.phonemize(text, lang=lang)

        # Split phonemes into batches
        batched_phonemes = self._split_phonemes(phonemes)

        for phoneme_batch in batched_phonemes:
            audio_part, sample_rate = self._create_audio_internal(
                phoneme_batch, voice_style, speed
            )
            # Trim silence
            audio_part, _ = trim_audio(audio_part)
            yield audio_part, sample_rate, phoneme_batch

    def close(self) -> None:
        """Clean up resources."""
        if self._voice_db is not None:
            self._voice_db.close()
            self._voice_db = None


# Language code mapping for kokoro-onnx
LANG_CODE_TO_ONNX = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt",  # Portuguese
    "z": "zh",  # Chinese
}


def is_chinese_language(lang: str) -> bool:
    """Check if language code is Chinese.

    Args:
        lang: Language code (e.g., 'zh', 'cmn', 'zh-cn')

    Returns:
        True if language is Chinese, False otherwise
    """
    lang_lower = lang.lower().strip()
    return lang_lower in ["zh", "cmn", "zh-cn", "zh-tw", "zh-hans", "zh-hant"]


def get_onnx_lang_code(ttsforge_lang: str) -> str:
    """Convert ttsforge language code to kokoro-onnx language code."""
    return LANG_CODE_TO_ONNX.get(ttsforge_lang, "en-us")
