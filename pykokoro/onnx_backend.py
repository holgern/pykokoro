"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

import asyncio
import io
import logging
import os
import re
import sqlite3
import urllib.request
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download

from .audio_generator import AudioGenerator
from .onnx_session import OnnxSessionManager
from .phonemes import PhonemeSegment
from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .trim import trim as trim_audio
from .utils import get_user_cache_path
from .voice_manager import VoiceBlend, VoiceManager

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

# Model variant type (for GitHub and HuggingFace sources)
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

# Note: Both HF v1.0 and v1.1-zh use the same filename convention
# (MODEL_QUALITY_FILES_HF)

# Backward compatibility
MODEL_QUALITY_FILES = MODEL_QUALITY_FILES_HF

# HuggingFace repositories for models and voices (onnx-community)
HF_REPO_V1_0 = "onnx-community/Kokoro-82M-v1.0-ONNX"
HF_REPO_V1_1_ZH = "onnx-community/Kokoro-82M-v1.1-zh-ONNX"

# HuggingFace repositories for configs (hexgrad)
HF_CONFIG_REPO_V1_0 = "hexgrad/Kokoro-82M"
HF_CONFIG_REPO_V1_1_ZH = "hexgrad/Kokoro-82M-v1.1-zh"

# Subfolders and filenames within HuggingFace repos
HF_MODEL_SUBFOLDER = "onnx"
HF_VOICES_SUBFOLDER = "voices"
HF_CONFIG_FILENAME = "config.json"

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

# All available voice names for v1.0 (54 voices - English/multilingual)
# Used by both HuggingFace and GitHub sources
# These are used for downloading individual voice files from HuggingFace
VOICE_NAMES_V1_0 = [
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

# Complete voice list for v1.1-zh (103 voices - Chinese)
# Used by both HuggingFace and GitHub sources
VOICE_NAMES_V1_1_ZH = [
    "af_maple",
    "af_sol",
    "bf_vale",
    "zf_001",
    "zf_002",
    "zf_003",
    "zf_004",
    "zf_005",
    "zf_006",
    "zf_007",
    "zf_008",
    "zf_017",
    "zf_018",
    "zf_019",
    "zf_021",
    "zf_022",
    "zf_023",
    "zf_024",
    "zf_026",
    "zf_027",
    "zf_028",
    "zf_032",
    "zf_036",
    "zf_038",
    "zf_039",
    "zf_040",
    "zf_042",
    "zf_043",
    "zf_044",
    "zf_046",
    "zf_047",
    "zf_048",
    "zf_049",
    "zf_051",
    "zf_059",
    "zf_060",
    "zf_067",
    "zf_070",
    "zf_071",
    "zf_072",
    "zf_073",
    "zf_074",
    "zf_075",
    "zf_076",
    "zf_077",
    "zf_078",
    "zf_079",
    "zf_083",
    "zf_084",
    "zf_085",
    "zf_086",
    "zf_087",
    "zf_088",
    "zf_090",
    "zf_092",
    "zf_093",
    "zf_094",
    "zf_099",
    "zm_009",
    "zm_010",
    "zm_011",
    "zm_012",
    "zm_013",
    "zm_014",
    "zm_015",
    "zm_016",
    "zm_020",
    "zm_025",
    "zm_029",
    "zm_030",
    "zm_031",
    "zm_033",
    "zm_034",
    "zm_035",
    "zm_037",
    "zm_041",
    "zm_045",
    "zm_050",
    "zm_052",
    "zm_053",
    "zm_054",
    "zm_055",
    "zm_056",
    "zm_057",
    "zm_058",
    "zm_061",
    "zm_062",
    "zm_063",
    "zm_064",
    "zm_065",
    "zm_066",
    "zm_068",
    "zm_069",
    "zm_080",
    "zm_081",
    "zm_082",
    "zm_089",
    "zm_091",
    "zm_095",
    "zm_096",
    "zm_097",
    "zm_098",
    "zm_100",
]

# Backward compatibility alias
VOICE_NAMES = VOICE_NAMES_V1_0

# Voice name documentation by language/variant
# These voices are dynamically loaded from the model's voices.bin file
# The actual available voices may vary depending on the model source and variant
VOICE_NAMES_BY_VARIANT = {
    "huggingface-v1.0": VOICE_NAMES_V1_0,  # All voices (multi-language)
    "huggingface-v1.1-zh": VOICE_NAMES_V1_1_ZH,  # Chinese-specific voices
    "github-v1.0": VOICE_NAMES_V1_0,  # Same as HuggingFace (multi-language)
    "github-v1.1-zh": VOICE_NAMES_V1_1_ZH,  # Chinese-specific voices
}


# =============================================================================
# Path helper functions
# =============================================================================


def get_model_dir(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get directory for model files.

    Returns: ~/.cache/pykokoro/models/{source}/{variant}/

    Args:
        source: Model source (huggingface or github)
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to model directory
    """
    model_dir = get_user_cache_path("models") / source / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_voices_dir(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get directory for voice files.

    Returns: ~/.cache/pykokoro/voices/{source}/{variant}/

    Args:
        source: Model source (huggingface or github)
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to voices directory
    """
    voices_dir = get_user_cache_path("voices") / source / variant
    voices_dir.mkdir(parents=True, exist_ok=True)
    return voices_dir


def get_config_path(variant: ModelVariant = DEFAULT_MODEL_VARIANT) -> Path:
    """
    Get path to config file (shared across sources for same variant).

    Returns: ~/.cache/pykokoro/config/{variant}/config.json

    Config files are downloaded from hexgrad repos and shared between
    HuggingFace and GitHub sources for the same variant.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to config file
    """
    config_dir = get_user_cache_path("config") / variant
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / HF_CONFIG_FILENAME


def get_voices_bin_path() -> Path:
    """Get the path to the combined voices.bin.npz file."""
    return get_user_cache_path() / "voices.bin.npz"


def get_model_path(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get full path to a specific model file.

    Args:
        quality: Model quality/quantization level
        source: Model source (huggingface or github)
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to model file

    Raises:
        ValueError: If quality is not available for the source/variant combination
    """
    model_dir = get_model_dir(source, variant)

    # Get appropriate filename mapping based on source and variant
    if source == "huggingface":
        # Both v1.0 and v1.1-zh use same filename convention
        quality_files = MODEL_QUALITY_FILES_HF
    elif source == "github":
        if variant == "v1.0":
            quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
        else:  # v1.1-zh
            quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
    else:
        raise ValueError(f"Unknown source: {source}")

    # Get filename for quality
    if quality not in quality_files:
        available = ", ".join(quality_files.keys())
        raise ValueError(
            f"Quality '{quality}' not available for {source}/{variant}. "
            f"Available: {available}"
        )

    filename = quality_files[quality]

    # HuggingFace models are stored in onnx/ subdirectory
    if source == "huggingface":
        return model_dir / HF_MODEL_SUBFOLDER / filename

    return model_dir / filename


def get_voice_path(
    voice_name: str,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """Get the full path to an individual voice file."""
    return get_voices_dir(source, variant) / f"{voice_name}.bin"


# =============================================================================
# Download check functions
# =============================================================================


def is_config_downloaded(variant: ModelVariant = DEFAULT_MODEL_VARIANT) -> bool:
    """Check if config.json is downloaded for a specific variant.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

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
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download config.json from hexgrad HuggingFace repository.

    Config files are downloaded from hexgrad repos and stored in a shared
    location used by both HuggingFace and GitHub sources.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded config file

    Note:
        - v1.0 config from: hexgrad/Kokoro-82M
        - v1.1-zh config from: hexgrad/Kokoro-82M-v1.1-zh
    """
    config_path = get_config_path(variant)

    if config_path.exists() and not force:
        logger.debug(f"Config already exists: {config_path}")
        return config_path

    # Select hexgrad repo based on variant
    if variant == "v1.0":
        repo_id = HF_CONFIG_REPO_V1_0  # hexgrad/Kokoro-82M
    elif variant == "v1.1-zh":
        repo_id = HF_CONFIG_REPO_V1_1_ZH  # hexgrad/Kokoro-82M-v1.1-zh
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger.info(f"Downloading config for {variant} from {repo_id}")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=HF_CONFIG_FILENAME,
        cache_dir=None,
        local_dir=config_path.parent,
        local_dir_use_symlinks=False,
    )

    return Path(downloaded_path)


def load_vocab_from_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> dict[str, int]:
    """Load vocabulary from variant-specific config.json.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

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
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download model from HuggingFace (onnx-community repos).

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If quality is not available

    Note:
        - v1.0 from: onnx-community/Kokoro-82M-v1.0-ONNX
        - v1.1-zh from: onnx-community/Kokoro-82M-v1.1-zh-ONNX
    """
    # Select onnx-community repo based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Check if quality is available (both variants use same filenames)
    if quality not in MODEL_QUALITY_FILES_HF:
        available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
        raise ValueError(f"Quality '{quality}' not available. Available: {available}")

    filename = MODEL_QUALITY_FILES_HF[quality]
    remote_path = f"{HF_MODEL_SUBFOLDER}/{filename}"

    # Use new path structure
    model_dir = get_model_dir(source="huggingface", variant=variant)
    local_path = model_dir / filename

    if local_path.exists() and not force:
        logger.debug(f"Model already exists: {local_path}")
        return local_path

    logger.info(f"Downloading {variant} model ({quality}) from {repo_id}")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        cache_dir=None,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )

    return Path(downloaded_path)


def download_voice(
    voice_name: str,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download a single voice file from HuggingFace.

    Args:
        voice_name: Name of the voice to download
        variant: Model variant (v1.0 or v1.1-zh)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voice file
    """
    # Select repo based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    filename = f"{voice_name}.bin"
    remote_path = f"{HF_VOICES_SUBFOLDER}/{filename}"

    # Use new path structure
    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)
    local_path = voices_dir / filename

    if local_path.exists() and not force:
        logger.debug(f"Voice already exists: {local_path}")
        return local_path

    logger.info(f"Downloading voice {voice_name} for {variant}")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        cache_dir=None,
        local_dir=voices_dir,
        local_dir_use_symlinks=False,
    )

    return Path(downloaded_path)


def download_all_voices(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """
    Download all voice files from HuggingFace for a specific variant.

    Downloads individual .bin files and combines them into voices.bin.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        progress_callback: Optional callback(filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Path to voices directory

    Note:
        - v1.0: 54 voices from onnx-community/Kokoro-82M-v1.0-ONNX
        - v1.1-zh: 103 voices from onnx-community/Kokoro-82M-v1.1-zh-ONNX
    """
    # Select repo and voice list based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
        voice_names = VOICE_NAMES_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
        voice_names = VOICE_NAMES_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)

    voices_bin_path = voices_dir / "voices.bin.npz"

    # If voices.bin.npz already exists and not forcing, return early
    if voices_bin_path.exists() and not force:
        logger.info(f"voices.bin.npz already exists at {voices_bin_path}")
        return voices_dir

    # Download individual voice files (.bin format from HuggingFace)
    total = len(voice_names)
    downloaded_files = []

    for idx, voice_name in enumerate(voice_names):
        if progress_callback:
            progress_callback(voice_name, idx, total)

        voice_path = voices_dir / f"{voice_name}.bin"

        # Download if not exists or force
        if not voice_path.exists() or force:
            try:
                # Download to cache, then copy to voices_dir
                # to avoid subdirectory issues
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{HF_VOICES_SUBFOLDER}/{voice_name}.bin",
                )
                # Copy from HF cache to our voices_dir
                import shutil

                shutil.copy(downloaded_path, voice_path)
                logger.info(f"Downloaded {voice_name}.bin")
                downloaded_files.append(voice_name)
            except Exception as e:
                logger.warning(f"Failed to download {voice_name}.bin: {e}")
                continue
        else:
            downloaded_files.append(voice_name)

    # Load and combine all voices into a single .npz file (voices.bin.npz)
    if downloaded_files:
        logger.info(f"Combining {len(downloaded_files)} voices into voices.bin.npz")
        voices_data = {}

        for voice_name in downloaded_files:
            voice_path = voices_dir / f"{voice_name}.bin"
            try:
                # HuggingFace .bin files are raw float32 arrays
                voice_data = np.fromfile(str(voice_path), dtype=np.float32)
                # Reshape to match expected format: (N, 1, 256) where N = len / 256
                # Most voices are 131072 floats = 512 * 1 * 256
                voice_data = voice_data.reshape(-1, 1, 256)
                voices_data[voice_name] = voice_data
            except Exception as e:
                logger.warning(f"Failed to load {voice_name}.bin: {e}")

        if voices_data:
            np.savez(str(voices_bin_path), **voices_data)
            logger.info(
                f"Created combined voices.bin.npz with {len(voices_data)} voices"
            )

    return voices_dir


def download_all_models(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download config, model, and all voice files for HuggingFace source.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download config
    if progress_callback:
        progress_callback("config.json", 0, 3)
    paths["config.json"] = download_config(variant=variant, force=force)

    # Download model
    if progress_callback:
        progress_callback("model", 1, 3)
    model_path = download_model(variant=variant, quality=quality, force=force)
    paths[model_path.name] = model_path

    # Download all voices
    if progress_callback:
        progress_callback("voices", 2, 3)
    voices_dir = download_all_voices(
        variant=variant, progress_callback=None, force=force
    )
    paths["voices"] = voices_dir

    if progress_callback:
        progress_callback("complete", 3, 3)

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

    # Use new path structure
    model_dir = get_model_dir(source="github", variant=variant)
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

    # Use new path structure
    voices_dir = get_voices_dir(source="github", variant=variant)
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
            model_variant: Model variant ("v1.0", "v1.1-zh")
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
            # Both v1.0 and v1.1-zh use same filename convention for HuggingFace
            if resolved_quality not in MODEL_QUALITY_FILES_HF:
                available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
                raise ValueError(
                    f"Quality '{resolved_quality}' not available for "
                    f"HuggingFace {model_variant}. Available qualities: {available}"
                )

        self._model_quality: ModelQuality = resolved_quality

        # Resolve paths
        if model_path is None:
            model_path = get_model_path(
                quality=self._model_quality, source=model_source, variant=model_variant
            )

        if voices_path is None:
            if model_source == "huggingface":
                # HuggingFace uses voices.bin.npz for both variants
                voices_path = (
                    get_voices_dir("huggingface", model_variant) / "voices.bin.npz"
                )
            elif model_source == "github":
                # GitHub uses variant-specific filenames
                if model_variant == "v1.0":
                    filename = GITHUB_VOICES_FILENAME_V1_0
                else:  # v1.1-zh
                    filename = GITHUB_VOICES_FILENAME_V1_1_ZH
                voices_path = get_voices_dir("github", model_variant) / filename

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

        # For GitHub models or v1.1-zh, load variant-specific vocab from config
        if self._model_source == "github" or self._model_variant == "v1.1-zh":
            return load_vocab_from_config(self._model_variant)

        # For HuggingFace v1.0 or default, use standard vocab
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
                download_model(variant=self._model_variant, quality=self._model_quality)

        # Download voices if needed
        if not self._voices_path.exists():
            if self._model_source == "github":
                download_voices_github(variant=self._model_variant)
            else:  # huggingface
                download_all_voices(variant=self._model_variant)

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

    def _init_kokoro(self) -> None:
        """Initialize the ONNX session and load voices."""
        if self._session is not None:
            return

        self._ensure_models()

        # Use OnnxSessionManager to create session
        session_manager = OnnxSessionManager(
            provider=self._provider,
            use_gpu=self._use_gpu,
            session_options=self._session_options,
            provider_options=self._provider_options,
        )
        self._session = session_manager.create_session(model_path=self._model_path)

        # Use VoiceManager to load voices
        voice_manager = VoiceManager(
            model_source=self._model_source,
        )
        voice_manager.load_voices(voices_path=self._voices_path)
        self._voices_data = voice_manager._voices_data

        # Create AudioGenerator
        self._audio_generator = AudioGenerator(
            session=self._session,
            tokenizer=self.tokenizer,
            model_source=self._model_source,
        )

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
                # Update model path using helper function
                model_dir = get_model_dir(source="github", variant=resolved_variant)

                if resolved_variant == "v1.0":
                    quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
                else:  # v1.1-zh
                    quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH

                filename = quality_files[self._model_quality]
                self._model_path = model_dir / filename

                # Update voices path using helper function
                voices_dir = get_voices_dir(source="github", variant=resolved_variant)

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
            audio_part, _ = self._audio_generator.generate_from_phonemes(
                batch, voice_style, speed
            )
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
                audio_part, _ = self._audio_generator.generate_from_phonemes(
                    batch, voice_style, speed
                )
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
                    None,
                    self._audio_generator.generate_from_phonemes,
                    phoneme_batch,
                    voice_style,
                    speed,
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
            audio_part, sample_rate = self._audio_generator.generate_from_phonemes(
                phoneme_batch, voice_style, speed
            )
            # Trim silence
            audio_part, _ = trim_audio(audio_part)
            yield audio_part, sample_rate, phoneme_batch

    # Delegate methods for backward compatibility with tests
    def _split_phonemes(self, phonemes: str) -> list[str]:
        """Delegate to AudioGenerator.split_phonemes (backward compatibility)."""
        self._init_kokoro()
        return self._audio_generator.split_phonemes(phonemes)

    def _generate_from_phoneme_batches(
        self,
        batches: list[str],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Delegate to AudioGenerator (backward compatibility)."""
        self._init_kokoro()
        return self._audio_generator.generate_from_phoneme_batches(
            batches, voice_style, speed, trim_silence
        )

    def _generate_from_segments(
        self,
        segments: list["PhonemeSegment"],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Delegate to AudioGenerator (backward compatibility)."""
        self._init_kokoro()
        return self._audio_generator.generate_from_segments(
            segments, voice_style, speed, trim_silence
        )

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
